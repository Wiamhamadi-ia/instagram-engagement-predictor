"""Entraîne le modèle final de la phase 4b et exporte ce dont l'app a besoin.

Deux sorties, toutes deux légères et versionnées :
  models/app/artefacts.joblib   moyennes de centrage, ACP, modèle Ridge
  models/app/reference.json     métriques honnêtes, seuils, agrégats de la phase 2

Usage:
    python -m src.models.export_app
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import src.models.train_relatif as M

SORTIE = M.RACINE / "models/app"
N_COMPOSANTES = 40


def modele():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def predictions_hors_echantillon(df, tab, nlp_c, cv_c, y, groupes):
    """Prédictions de validation croisée, pour des seuils et une AUC non optimistes.

    Le centrage appliqué au test est celui de l'app — moyenne globale du train —
    et non le centrage par compte, indisponible pour un nouvel utilisateur.
    """
    oof = np.full(len(y), np.nan)
    par_fold = []
    brut_nlp = df[[c for c in df.columns if c.startswith("nlp_")]].values
    brut_cv = df[[c for c in df.columns if c.startswith("cv_")]].values

    for i_tr, i_te in GroupKFold(n_splits=5).split(tab, y, groupes):
        acp_n = PCA(N_COMPOSANTES, random_state=0).fit(nlp_c[i_tr])
        acp_c = PCA(N_COMPOSANTES, random_state=0).fit(cv_c[i_tr])
        m = modele().fit(
            np.hstack([tab[i_tr], acp_n.transform(nlp_c[i_tr]), acp_c.transform(cv_c[i_tr])]),
            y[i_tr])
        te = np.hstack([
            tab[i_te],
            acp_n.transform(brut_nlp[i_te] - brut_nlp[i_tr].mean(0)),
            acp_c.transform(brut_cv[i_te] - brut_cv[i_tr].mean(0)),
        ])
        oof[i_te] = m.predict(te)
        par_fold.append(float(roc_auc_score((y[i_te] > 0).astype(int), oof[i_te])))
    return oof, par_fold


def agregats_phase2():
    """Courbes descriptives de la phase 2, pour la page d'exploration de l'app."""
    d = pd.read_csv(M.PROC / "dataset_clean.csv", parse_dates=["timestamp"])
    u = d[d.utilisable_entrainement].copy()
    u["rel"] = u.interactions / u.groupby("compte").interactions.transform("median")

    heure = u.groupby("heure").agg(n=("id", "count"), rel=("rel", "median"))
    tranches = pd.cut(u.nb_hashtags, [-1, 0, 5, 10, 20, 10_000],
                      labels=["aucun", "1 à 5", "6 à 10", "11 à 20", "21 et +"])
    ht = u.groupby(tranches, observed=True).agg(n=("id", "count"), rel=("rel", "median"))
    jours = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    jour = u.groupby("nom_jour").rel.median().reindex(jours)

    return {
        "heure": {"n": heure.n.tolist(), "relatif": heure.rel.round(4).tolist()},
        "hashtags": {"libelles": ht.index.tolist(), "n": ht.n.tolist(),
                     "relatif": ht.rel.round(4).tolist()},
        "jour": {"relatif": jour.round(4).tolist()},
        "heure_pic": int(heure[heure.n >= 60].rel.idxmax()),
    }


def main():
    df, perdus = M.construire_cible(M.charger())
    cols_nlp = [c for c in df.columns if c.startswith("nlp_")]
    cols_cv = [c for c in df.columns if c.startswith("cv_")]

    nlp_c = M.centrer_causal(df[cols_nlp].values, df.compte)
    cv_c = M.centrer_causal(df[cols_cv].values, df.compte)

    tab = df[M.TABULAIRES].copy()
    tab["jours_depuis_dernier_post"] = tab.jours_depuis_dernier_post.fillna(
        tab.jours_depuis_dernier_post.median())
    mediane_gap = float(tab.jours_depuis_dernier_post.median())
    tab = tab.values
    y, groupes = df.cible_relative.values, df.compte.values

    print("Validation croisee pour les seuils et l'AUC affichee par l'app...")
    oof, auc_folds = predictions_hors_echantillon(df, tab, nlp_c, cv_c, y, groupes)
    auc_app = float(np.mean(auc_folds))
    seuils = [float(np.quantile(oof, q)) for q in (1 / 3, 2 / 3)]
    print(f"  AUC dans les conditions de l'app : {auc_app:.3f}"
          f" +/-{np.std(auc_folds):.3f} (moyenne sur 5 folds)")
    print(f"  seuils des tiers : {seuils[0]:+.4f} / {seuils[1]:+.4f}")

    # Modèle final : tout le jeu, centrage causal comme à l'entraînement
    acp_nlp = PCA(N_COMPOSANTES, random_state=0).fit(nlp_c)
    acp_cv = PCA(N_COMPOSANTES, random_state=0).fit(cv_c)
    X = np.hstack([tab, acp_nlp.transform(nlp_c), acp_cv.transform(cv_c)])
    final = modele().fit(X, y)

    SORTIE.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "modele": final,
        "acp_nlp": acp_nlp,
        "acp_cv": acp_cv,
        # Références de centrage pour un post isolé, sans historique d'embeddings
        "moyenne_nlp": df[cols_nlp].values.mean(0),
        "moyenne_cv": df[cols_cv].values.mean(0),
        "colonnes_tabulaires": M.TABULAIRES,
        "mediane_gap": mediane_gap,
    }, SORTIE / "artefacts.joblib")

    (SORTIE / "reference.json").write_text(json.dumps({
        "auc_conditions_app": auc_app,
        "auc_par_fold": [round(a, 4) for a in auc_folds],
        "auc_centrage_par_compte": 0.530,
        "seuils_tiers": seuils,
        "entrainement": {
            "posts": len(df), "comptes": int(df.compte.nunique()),
            "posts_ecartes": perdus,
            "followers_min": 255, "followers_max": 11209,
            "part_video": 0.562,
            "ecart_type_cible": float(df.cible_relative.std()),
        },
        "phase2": agregats_phase2(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    taille = sum(f.stat().st_size for f in SORTIE.iterdir()) // 1024
    print(f"-> {SORTIE} ({taille} Ko)")


if __name__ == "__main__":
    main()
