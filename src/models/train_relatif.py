"""Phase 4b : cible relative au compte, embeddings centrés puis réduits.

La phase 4 a échoué pour une raison identifiée : 44 % de la variance à prédire se
situe entre les comptes, et 21 comptes d'entraînement ne suffisent pas à apprendre
à généraliser de l'un à l'autre. Ce script change donc la question posée.

Au lieu de « combien de réactions ce post va-t-il recevoir ? », on demande
« ce post fera-t-il mieux ou moins bien que l'ordinaire de son compte ? ».

Usage:
    python -m src.models.train_relatif
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RACINE = Path(__file__).resolve().parents[2]
PROC = RACINE / "data/processed"
RESULTATS = PROC / "resultats_relatif.json"

FENETRE = 20          # nombre de posts précédents pris en compte
MIN_HISTORIQUE = 10   # en-deçà, le niveau habituel n'est pas assez fiable
N_COMPOSANTES = 40    # par bloc d'embeddings
N_FOLDS = 5

# `followers` et `theme` sont constants à l'intérieur d'un compte : avec une cible
# relative au compte, ils ne peuvent plus rien expliquer.
TABULAIRES = ["heure_sin", "heure_cos", "jours_depuis_dernier_post",
              "hashtags_tranche", "type_image", "type_carrousel", "type_video",
              "caption_vide", "image_manquante"]


def charger():
    f = pd.read_csv(PROC / "features.csv")
    d = pd.read_csv(PROC / "dataset_clean.csv",
                    usecols=["id", "compte", "timestamp", "interactions"],
                    parse_dates=["timestamp"])
    df = f.merge(d, on="id")
    df = df[df.utilisable_entrainement].sort_values(["compte", "timestamp"])
    return df.reset_index(drop=True)


def construire_cible(df):
    """Niveau habituel = médiane des posts PRÉCÉDENTS du même compte.

    Le `shift(1)` est ce qui rend la cible utilisable en prédiction : au moment de
    publier, on ne connaît que le passé. Sans lui, le post courant entrerait dans le
    calcul de son propre point de comparaison.
    """
    niveau = df.groupby("compte").interactions.transform(
        lambda s: s.shift(1).rolling(FENETRE, min_periods=MIN_HISTORIQUE).median())
    df = df.assign(niveau_habituel=niveau)
    avant = len(df)
    df = df[df.niveau_habituel.notna()].reset_index(drop=True)
    df["cible_relative"] = np.log1p(df.interactions) - np.log1p(df.niveau_habituel)
    return df, avant - len(df)


def centrer_global(X, comptes):
    """Écart au style moyen du compte, calculé sur tous ses posts.

    À faire AVANT la PCA : sur des vecteurs non centrés, les premières composantes
    captureraient surtout « de quel compte vient ce post », qui est justement
    l'information dont la cible relative nous débarrasse.

    Réserve : la moyenne inclut les posts postérieurs, information indisponible au
    moment de publier. Comparer avec `centrer_causal` pour mesurer ce que ce
    regard vers le futur apporte.
    """
    moyennes = pd.DataFrame(X).groupby(comptes.values).transform("mean").values
    return X - moyennes


def centrer_causal(X, comptes):
    """Écart à la moyenne des posts PRÉCÉDENTS du même compte.

    Version utilisable en production : au moment de publier, seul le passé existe.
    Le premier post de chaque compte est centré sur un vecteur nul, faute de passé.
    """
    sortie = np.empty_like(X)
    for compte in pd.unique(comptes):
        masque = (comptes == compte).values
        bloc = X[masque]
        cumul = np.cumsum(bloc, axis=0)
        effectifs = np.arange(1, len(bloc)).reshape(-1, 1)
        moyenne_passee = np.vstack([np.zeros((1, X.shape[1])), cumul[:-1] / effectifs])
        sortie[masque] = bloc - moyenne_passee
    return sortie


CENTRAGES = {"global": centrer_global, "causal": centrer_causal}


def metriques(y, pred, comptes):
    """Trois angles : classement interne, décision binaire, erreur brute."""
    rangs = []
    for c in np.unique(comptes):
        m = comptes == c
        if m.sum() >= 5 and np.std(pred[m]) > 0:
            rho = stats.spearmanr(y[m], pred[m]).statistic
            if not np.isnan(rho):
                rangs.append(rho)
    binaire = (y > 0).astype(int)
    auc = float(roc_auc_score(binaire, pred)) if 0 < binaire.mean() < 1 else float("nan")
    return {
        "spearman_intra_compte": float(np.mean(rangs)) if rangs else float("nan"),
        "auc": auc,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "n_comptes_evalues": len(rangs),
    }


def modeles(seed=0):
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=seed)),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbosity=0),
        "LightGBM": LGBMRegressor(
            n_estimators=400, num_leaves=15, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_samples=20, n_jobs=-1,
            random_state=seed, verbose=-1),
    }


def evaluer_cv(df, centrage="global", n_composantes=N_COMPOSANTES,
               n_folds=N_FOLDS, seed=0):
    cols_nlp = [c for c in df.columns if c.startswith("nlp_")]
    cols_cv = [c for c in df.columns if c.startswith("cv_")]

    # Le centrage est une opération interne à chaque compte : il ne fait circuler
    # aucune information d'un compte à l'autre, donc il peut précéder le découpage.
    centrer = CENTRAGES[centrage]
    nlp_c = centrer(df[cols_nlp].values, df.compte)
    cv_c = centrer(df[cols_cv].values, df.compte)

    tab = df[TABULAIRES].copy()
    tab["jours_depuis_dernier_post"] = tab.jours_depuis_dernier_post.fillna(
        tab.jours_depuis_dernier_post.median())
    tab = tab.values
    y, groupes = df.cible_relative.values, df.compte.values

    resultats = {nom: [] for nom in modeles()}
    resultats["Baseline : niveau habituel"] = []
    variance_pca = []

    for k, (i_tr, i_te) in enumerate(GroupKFold(n_splits=n_folds).split(tab, y, groupes)):
        # La PCA est réajustée dans chaque fold, sur le train seul
        blocs_tr, blocs_te, explique = [tab[i_tr]], [tab[i_te]], {}
        for nom, bloc in [("texte", nlp_c), ("image", cv_c)]:
            acp = PCA(n_components=n_composantes, random_state=seed).fit(bloc[i_tr])
            blocs_tr.append(acp.transform(bloc[i_tr]))
            blocs_te.append(acp.transform(bloc[i_te]))
            explique[nom] = float(acp.explained_variance_ratio_.sum())
        variance_pca.append(explique)

        X_tr, X_te = np.hstack(blocs_tr), np.hstack(blocs_te)
        y_tr, y_te, g_te = y[i_tr], y[i_te], groupes[i_te]

        # Prédire 0, c'est parier que chaque post fera exactement comme d'habitude
        resultats["Baseline : niveau habituel"].append(
            metriques(y_te, np.zeros_like(y_te), g_te))

        for nom, modele in modeles(seed).items():
            modele.fit(X_tr, y_tr)
            resultats[nom].append(metriques(y_te, modele.predict(X_te), g_te))

        print(f"    fold {k + 1}/{n_folds} : {len(i_te)} posts, "
              f"{len(np.unique(g_te))} comptes | variance ACP "
              f"texte {explique['texte']:.0%} image {explique['image']:.0%}")

    resume = {}
    for nom, folds in resultats.items():
        resume[nom] = {
            cle: {"moyenne": float(np.nanmean([f[cle] for f in folds])),
                  "ecart_type": float(np.nanstd([f[cle] for f in folds])),
                  "par_fold": [round(f[cle], 4) for f in folds]}
            for cle in ("spearman_intra_compte", "auc", "rmse")
        }
    return resume, variance_pca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composantes", type=int, default=N_COMPOSANTES)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    brut = charger()
    df, perdus = construire_cible(brut)
    print(f"{len(brut)} posts exploitables -> {len(df)} avec historique suffisant")
    print(f"  {perdus} posts ecartes ({perdus / len(brut) * 100:.1f} %) : "
          f"moins de {MIN_HISTORIQUE} posts anterieurs")
    print(f"  {df.compte.nunique()} comptes conserves")
    print(f"  cible : moyenne {df.cible_relative.mean():+.3f}, "
          f"ecart-type {df.cible_relative.std():.3f}, "
          f"{(df.cible_relative > 0).mean():.1%} au-dessus du niveau habituel\n")

    tout = {}
    for centrage in ("global", "causal"):
        print(f"[centrage {centrage}]")
        resume, variance = evaluer_cv(df, centrage, args.composantes, args.folds)
        tout[centrage] = {"resultats": resume, "variance_expliquee_pca": variance}

        print()
        print(f"  {'modele':28} {'Spearman intra':>17} {'AUC':>15} {'RMSE':>15}")
        for nom, m in resume.items():
            rho = m["spearman_intra_compte"]
            rho_txt = ("      n/a       " if np.isnan(rho["moyenne"])
                       else f"{rho['moyenne']:+.3f} +/-{rho['ecart_type']:.3f}")
            print(f"  {nom:28} {rho_txt:>17} "
                  f"{m['auc']['moyenne']:.3f} +/-{m['auc']['ecart_type']:.3f}  "
                  f"{m['rmse']['moyenne']:.3f} +/-{m['rmse']['ecart_type']:.3f}")
        print()

    ecart = {nom: tout["global"]["resultats"][nom]["auc"]["moyenne"]
                  - tout["causal"]["resultats"][nom]["auc"]["moyenne"]
             for nom in tout["global"]["resultats"]}
    print("Ce que le centrage global gagne en regardant vers le futur (AUC) :")
    for nom, v in ecart.items():
        if nom.startswith("Baseline"):
            continue
        print(f"  {nom:28} {v:+.3f}")

    RESULTATS.write_text(json.dumps({
        "parametres": {"fenetre": FENETRE, "min_historique": MIN_HISTORIQUE,
                       "composantes_par_bloc": args.composantes, "folds": args.folds},
        "donnees": {"posts_exploitables": len(brut), "posts_retenus": len(df),
                    "posts_perdus": perdus, "comptes": int(df.compte.nunique()),
                    "part_au_dessus": float((df.cible_relative > 0).mean()),
                    "ecart_type_cible": float(df.cible_relative.std())},
        "par_centrage": tout,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {RESULTATS}")


if __name__ == "__main__":
    main()
