"""Phase 4 : entraînement et comparaison des modèles.

Produit data/processed/resultats_modeles.json (lu par le notebook) et
models/modele_final.joblib.

Usage:
    python -m src.models.train_models
    python -m src.models.train_models --seeds 3   # plus rapide pour un essai
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

# Chemins ancrés à la racine du projet : le module est aussi importé depuis notebooks/
RACINE = Path(__file__).resolve().parents[2]
PROC = RACINE / "data/processed"
FEATURES = PROC / "features.csv"
DATASET = PROC / "dataset_clean.csv"
RESULTATS = PROC / "resultats_modeles.json"
MODELE = RACINE / "models/modele_final.joblib"

SEUIL_PETIT_COMPTE = 2000
SEEDS = [0, 1, 2, 3, 4]

# Colonnes qui ne sont pas des variables d'entrée. `followers` et
# `jours_depuis_dernier_post` sont exclus au profit de leurs versions
# standardisées sur le train, pour ne pas dupliquer l'information.
HORS_FEATURES = ["id", "groupe_split", "utilisable_entrainement", "log_interactions",
                 "followers", "jours_depuis_dernier_post"]


def charger():
    f = pd.read_csv(FEATURES)
    d = pd.read_csv(DATASET, usecols=["id", "compte", "followers", "interactions"])
    f = f.merge(d, on="id", suffixes=("", "_ref"))

    u = f[f.utilisable_entrainement].reset_index(drop=True)
    colonnes = [c for c in u.columns
                if c not in HORS_FEATURES + ["compte", "followers_ref", "interactions"]]

    train = u.groupe_split == "train"
    # Imputation du premier post de chaque compte : médiane du train uniquement
    mediane = u.loc[train, "jours_depuis_dernier_post_z"].median()
    u["jours_depuis_dernier_post_z"] = u["jours_depuis_dernier_post_z"].fillna(mediane)

    return u, colonnes, train.values, (u.groupe_split == "validation").values


def evaluer(y_vrai, y_pred):
    """Métriques en échelle log, puis reconverties en interactions réelles."""
    rmse_log = float(np.sqrt(mean_squared_error(y_vrai, y_pred)))
    mae_log = float(mean_absolute_error(y_vrai, y_pred))
    vrai, pred = np.expm1(y_vrai), np.expm1(np.clip(y_pred, 0, 20))
    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        # Une erreur de 0,7 en log signifie « faux d'un facteur e^0,7 = 2 »
        "facteur_erreur": float(np.exp(mae_log)),
        "mae_interactions": float(mean_absolute_error(vrai, pred)),
        "mediane_erreur_interactions": float(np.median(np.abs(vrai - pred))),
    }


def modeles(seed):
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "XGBoost": XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbosity=0),
        "LightGBM": LGBMRegressor(
            n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbose=-1,
            importance_type="gain"),
    }


def baselines(u, colonnes, tr, va):
    y_tr, y_va = u.log_interactions[tr], u.log_interactions[va]
    res = {}

    trivial = DummyRegressor(strategy="median").fit(u.loc[tr, colonnes], y_tr)
    res["Mediane du train"] = evaluer(y_va, trivial.predict(u.loc[va, colonnes]))

    lin = LinearRegression().fit(u.loc[tr, ["followers_z"]], y_tr)
    res["Regression sur followers"] = evaluer(y_va, lin.predict(u.loc[va, ["followers_z"]]))
    return res


def multi_seed(u, colonnes, tr, va, seeds):
    """Chaque modèle est entraîné avec plusieurs graines pour mesurer sa stabilité."""
    X_tr, y_tr = u.loc[tr, colonnes], u.log_interactions[tr]
    X_va, y_va = u.loc[va, colonnes], u.log_interactions[va]
    comptes_va = u.compte[va].values

    resultats, predictions = {}, {}
    for nom in modeles(0):
        scores, preds = [], []
        for seed in seeds:
            t0 = time.time()
            m = modeles(seed)[nom].fit(X_tr, y_tr)
            p = m.predict(X_va)
            preds.append(p)
            scores.append(evaluer(y_va, p))
            print(f"    {nom:14} seed {seed}  rmse={scores[-1]['rmse_log']:.4f}"
                  f"  ({time.time() - t0:.0f}s)")
        resultats[nom] = {
            cle: {"moyenne": float(np.mean([s[cle] for s in scores])),
                  "ecart_type": float(np.std([s[cle] for s in scores]))}
            for cle in scores[0]
        }
        predictions[nom] = np.mean(preds, axis=0)

    par_compte = {
        nom: erreur_par_compte(y_va.values, p, comptes_va)
        for nom, p in predictions.items()
    }
    return resultats, par_compte


def erreur_par_compte(y_vrai, y_pred, comptes):
    """La validation ne compte que 7 comptes : un seul peut porter toute l'erreur."""
    df = pd.DataFrame({"compte": comptes, "err": np.abs(y_vrai - y_pred),
                       "err2": (y_vrai - y_pred) ** 2})
    g = df.groupby("compte").agg(posts=("err", "size"), mae=("err", "mean"),
                                 sse=("err2", "sum"))
    g["part_erreur_totale"] = g.sse / g.sse.sum()
    return g.sort_values("part_erreur_totale", ascending=False).round(4).to_dict("index")


def tuning(u, colonnes, tr, nom, seed=0, n_essais=18):
    """Recherche aléatoire courte, validée par GroupKFold sur les comptes du train.

    Le réglage n'utilise jamais le jeu de validation : sinon la performance
    rapportée ensuite serait optimiste.
    """
    X, y = u.loc[tr, colonnes], u.log_interactions[tr]
    groupes = u.compte[tr]
    cv = GroupKFold(n_splits=5)
    rng = np.random.default_rng(seed)

    grilles = {
        "LightGBM": lambda: dict(
            n_estimators=int(rng.choice([300, 500, 800])),
            num_leaves=int(rng.choice([15, 31, 63])),
            learning_rate=float(rng.choice([0.02, 0.05, 0.1])),
            min_child_samples=int(rng.choice([10, 20, 40])),
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1,
            importance_type="gain"),
        "XGBoost": lambda: dict(
            n_estimators=int(rng.choice([300, 500, 800])),
            max_depth=int(rng.choice([4, 6, 8])),
            learning_rate=float(rng.choice([0.02, 0.05, 0.1])),
            min_child_weight=int(rng.choice([1, 5, 10])),
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbosity=0),
        "RandomForest": lambda: dict(
            n_estimators=int(rng.choice([300, 500])),
            max_depth=int(rng.choice([10, 20, 30])),
            min_samples_leaf=int(rng.choice([1, 2, 5])), n_jobs=-1),
    }
    classe = {"LightGBM": LGBMRegressor, "XGBoost": XGBRegressor,
              "RandomForest": RandomForestRegressor}[nom]

    essais = []
    for k in range(n_essais):
        params = grilles[nom]()
        scores = []
        for i_tr, i_te in cv.split(X, y, groupes):
            m = classe(random_state=0, **params).fit(X.iloc[i_tr], y.iloc[i_tr])
            scores.append(np.sqrt(mean_squared_error(y.iloc[i_te], m.predict(X.iloc[i_te]))))
        essais.append({"params": params, "rmse_cv": float(np.mean(scores)),
                       "ecart_cv": float(np.std(scores))})
        print(f"    essai {k + 1}/{n_essais}  rmse_cv={essais[-1]['rmse_cv']:.4f}")
    essais.sort(key=lambda e: e["rmse_cv"])
    return essais[0], essais


def comparer_petits_comptes(u, colonnes, tr, va, nom, params, seed=0):
    """Modèle entraîné sur tout, contre modèle entraîné sur les petits comptes seuls.

    Les deux sont évalués sur le MÊME sous-ensemble de validation (< 2 000 abonnés),
    sans quoi la comparaison n'aurait aucun sens.
    """
    classe = {"LightGBM": LGBMRegressor, "XGBoost": XGBRegressor,
              "RandomForest": RandomForestRegressor}[nom]
    petit = (u.followers_ref < SEUIL_PETIT_COMPTE).values
    va_petit = va & petit
    tr_petit = tr & petit

    complet = classe(random_state=seed, **params).fit(u.loc[tr, colonnes], u.log_interactions[tr])
    restreint = classe(random_state=seed, **params).fit(
        u.loc[tr_petit, colonnes], u.log_interactions[tr_petit])

    y = u.log_interactions[va_petit]
    return {
        "posts_train_complet": int(tr.sum()),
        "posts_train_restreint": int(tr_petit.sum()),
        "posts_validation_petits": int(va_petit.sum()),
        "comptes_train_restreint": int(u.compte[tr_petit].nunique()),
        "comptes_validation_petits": int(u.compte[va_petit].nunique()),
        "modele_complet": evaluer(y, complet.predict(u.loc[va_petit, colonnes])),
        "modele_restreint": evaluer(y, restreint.predict(u.loc[va_petit, colonnes])),
    }


def blocs_variables(colonnes):
    tab = [c for c in colonnes if not c.startswith(("nlp_", "cv_"))]
    nlp = [c for c in colonnes if c.startswith("nlp_")]
    cv = [c for c in colonnes if c.startswith("cv_")]
    return {
        "tabulaire seul": tab,
        "tabulaire + texte": tab + nlp,
        "tabulaire + image": tab + cv,
        "tout": colonnes,
    }


def ablation(u, colonnes, tr, va, nom, params, seed=0):
    """Chaque bloc de variables apporte-t-il quelque chose ?

    896 des 911 variables sont des embeddings. Avec seulement 2 651 exemples
    d'entraînement, elles peuvent diluer le signal au lieu de l'enrichir : ce test
    le mesure au lieu de le supposer.
    """
    classe = {"LightGBM": LGBMRegressor, "XGBoost": XGBRegressor,
              "RandomForest": RandomForestRegressor}[nom]
    y_tr, y_va = u.log_interactions[tr], u.log_interactions[va]
    res = {}
    for libelle, cols in blocs_variables(colonnes).items():
        m = classe(random_state=seed, **params).fit(u.loc[tr, cols], y_tr)
        res[libelle] = {"n_variables": len(cols),
                        **evaluer(y_va, m.predict(u.loc[va, cols]))}
        print(f"    {libelle:20} ({len(cols):3} var)  rmse={res[libelle]['rmse_log']:.4f}")
    return res


def importances(modele, colonnes, X_va, n=25):
    """Importance native, agrégée par bloc pour rester lisible avec 900 variables."""
    brute = pd.Series(modele.feature_importances_, index=colonnes)
    brute = brute / brute.sum()

    def bloc(c):
        if c.startswith("nlp_"):
            return "texte (embeddings)"
        if c.startswith("cv_"):
            return "image (embeddings)"
        if c.startswith("theme_"):
            return "theme"
        if c.startswith("type_"):
            return "format"
        return c

    par_bloc = brute.groupby(bloc).sum().sort_values(ascending=False)
    return {
        "par_bloc": par_bloc.round(4).to_dict(),
        "top_variables": brute.nlargest(n).round(4).to_dict(),
    }


def decomposition_variance(u):
    """Où se situe la variance que le modèle doit prédire ?

    Sert à interpréter les effets mesurés en phase 2 : ceux-ci étaient relatifs au
    compte, alors que le modèle prédit une valeur absolue.
    """
    y = u.log_interactions
    moyenne_compte = u.groupby("compte").log_interactions.transform("mean")
    totale = float(y.var())
    intra = float((y - moyenne_compte).var())
    residu = y - moyenne_compte
    cle_heure = u.heure_sin.round(4).astype(str) + "_" + u.heure_cos.round(4).astype(str)
    var_heure = float(residu.groupby(cle_heure).transform("mean").var())
    return {
        "variance_totale": totale,
        "entre_comptes": float(moyenne_compte.var()),
        "intra_compte": intra,
        "part_entre_comptes": float(moyenne_compte.var() / totale),
        "variance_expliquee_par_heure": var_heure,
        "part_heure_dans_intra": float(var_heure / intra),
        "part_heure_dans_totale": float(var_heure / totale),
    }


def importance_permutation(modele, X, y, colonnes, n_repetitions=5, seed=0):
    """Dégradation du RMSE quand on mélange un bloc de variables.

    Mesure la contribution réelle à la généralisation, là où l'importance native
    ne mesure que l'usage fait des variables pendant l'entraînement. Une valeur
    négative signifie que le bloc dégrade la prédiction sur la validation.
    """
    rng = np.random.default_rng(seed)
    reference = float(np.sqrt(mean_squared_error(y, modele.predict(X))))
    blocs = {
        "followers": ["followers_z"],
        "heure": ["heure_sin", "heure_cos"],
        "hashtags": ["hashtags_tranche"],
        "jours_depuis_dernier_post": ["jours_depuis_dernier_post_z"],
        "theme": [c for c in colonnes if c.startswith("theme_")],
        "format": [c for c in colonnes if c.startswith("type_")],
        "texte (embeddings)": [c for c in colonnes if c.startswith("nlp_")],
        "image (embeddings)": [c for c in colonnes if c.startswith("cv_")],
    }
    res = {"rmse_reference": reference}
    for nom, cols in blocs.items():
        ecarts = []
        for _ in range(n_repetitions):
            X2 = X.copy()
            for c in cols:
                X2[c] = rng.permutation(X2[c].values)
            ecarts.append(float(np.sqrt(mean_squared_error(y, modele.predict(X2)))) - reference)
        res[nom] = {"degradation_rmse": float(np.mean(ecarts)),
                    "ecart_type": float(np.std(ecarts))}
    return res


def coherence_phase2(imp_bloc, imp_brute):
    """La phase 2 a mesuré des effets réels : on vérifie qu'ils survivent au modèle."""
    attendus = {
        "followers_z": "taille du compte (rho = +0,37)",
        "heure_sin": "heure de publication (facteur 1,7)",
        "heure_cos": "heure de publication (facteur 1,7)",
        "hashtags_tranche": "hashtags 1-5 (+9 %)",
    }
    controle = {}
    for var, mesure in attendus.items():
        poids = float(imp_brute.get(var, 0.0))
        controle[var] = {
            "mesure_phase2": mesure,
            "importance": round(poids, 5),
            "statut": "anomalie" if poids < 0.001 else "coherent",
        }
    return controle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--essais", type=int, default=18)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]

    u, colonnes, tr, va = charger()
    print(f"{len(u)} posts exploitables | {len(colonnes)} variables")
    print(f"  train      : {tr.sum()} posts / {u.compte[tr].nunique()} comptes")
    print(f"  validation : {va.sum()} posts / {u.compte[va].nunique()} comptes")

    print("\n[1/6] Baselines")
    base = baselines(u, colonnes, tr, va)
    for nom, m in base.items():
        print(f"    {nom:26} rmse={m['rmse_log']:.4f}  mae={m['mae_log']:.4f}")

    print(f"\n[2/6] Trois candidats x {len(seeds)} seeds")
    candidats, par_compte = multi_seed(u, colonnes, tr, va, seeds)

    # Sélection : on pénalise l'instabilité plutôt que de suivre le seul RMSE moyen
    classement = sorted(candidats.items(),
                        key=lambda kv: kv[1]["rmse_log"]["moyenne"] + kv[1]["rmse_log"]["ecart_type"])
    retenu = classement[0][0]
    print(f"\n    -> retenu : {retenu}")

    print(f"\n[3/6] Tuning leger de {retenu} (GroupKFold sur les comptes du train)")
    meilleur, essais = tuning(u, colonnes, tr, retenu, n_essais=args.essais)
    print(f"    meilleurs parametres : {meilleur['params']}")

    classe = {"LightGBM": LGBMRegressor, "XGBoost": XGBRegressor,
              "RandomForest": RandomForestRegressor}[retenu]
    final = classe(random_state=0, **meilleur["params"]).fit(
        u.loc[tr, colonnes], u.log_interactions[tr])
    perf_final = evaluer(u.log_interactions[va], final.predict(u.loc[va, colonnes]))
    print(f"    validation : rmse={perf_final['rmse_log']:.4f} mae={perf_final['mae_log']:.4f}")

    print(f"\n[4/6] Echantillon complet vs comptes < {SEUIL_PETIT_COMPTE} abonnes")
    petits = comparer_petits_comptes(u, colonnes, tr, va, retenu, meilleur["params"])
    print(f"    complet   : rmse={petits['modele_complet']['rmse_log']:.4f}")
    print(f"    restreint : rmse={petits['modele_restreint']['rmse_log']:.4f}")

    print("\n[5/6] Ablation par bloc de variables")
    abl = ablation(u, colonnes, tr, va, retenu, meilleur["params"])

    print("\n[6/6] Importances")
    imp = importances(final, colonnes, u.loc[va, colonnes])
    brute = pd.Series(final.feature_importances_, index=colonnes)
    brute = brute / brute.sum()
    controle = coherence_phase2(imp["par_bloc"], brute.to_dict())
    for var, c in controle.items():
        print(f"    {var:20} {c['importance']:.5f}  {c['statut']}")

    perm = importance_permutation(final, u.loc[va, colonnes], u.log_interactions[va], colonnes)
    print("    importance par permutation (negatif = degrade la validation) :")
    for nom, v in perm.items():
        if nom != "rmse_reference":
            print(f"      {nom:26} {v['degradation_rmse']:+.4f}")
    variance = decomposition_variance(u)
    print(f"    l'heure explique {variance['part_heure_dans_totale'] * 100:.2f} %"
          f" de la variance totale a predire")

    MODELE.parent.mkdir(exist_ok=True)
    import joblib
    joblib.dump({"modele": final, "colonnes": colonnes, "params": meilleur["params"]}, MODELE)

    RESULTATS.write_text(json.dumps({
        "seeds": seeds,
        "n_variables": len(colonnes),
        "train": {"posts": int(tr.sum()), "comptes": int(u.compte[tr].nunique())},
        "validation": {"posts": int(va.sum()), "comptes": int(u.compte[va].nunique())},
        "baselines": base,
        "candidats": candidats,
        "erreur_par_compte": par_compte,
        "modele_retenu": retenu,
        "tuning": {"meilleur": meilleur, "essais": essais},
        "modele_final": perf_final,
        "petits_comptes": petits,
        "ablation": abl,
        "importances": imp,
        "coherence_phase2": controle,
        "importance_permutation": perm,
        "decomposition_variance": variance,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {RESULTATS}\n-> {MODELE}")


if __name__ == "__main__":
    main()
