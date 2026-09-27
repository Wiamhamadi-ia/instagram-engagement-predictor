"""Construit data/processed/features.csv à partir du jeu nettoyé.

Trois piliers : temporel, texte (sentence-transformers) et image (CLIP), plus les
variables de compte.

Usage:
    python -m src.features.build_features            # réutilise le cache d'embeddings
    python -m src.features.build_features --recalc   # recalcule tout

Les embeddings sont coûteux : ils sont mis en cache dans embeddings_nlp.npy et
embeddings_cv.npy, indexés par identifiant de post. Ajouter des comptes plus tard
ne recalcule que les posts nouveaux.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("data/processed")
SOURCE = PROC / "dataset_clean.csv"
SORTIE = PROC / "features.csv"
PARAMS = PROC / "scaler_params.json"
MEDIA = Path("data/raw/media")

# Variante multilingue de MiniLM : 29 % des légendes sont en écriture non latine
# (996 en arabe, dont 97 % du thème ecriture). Testé sur des phrases arabes de sens
# opposés, all-MiniLM-L6-v2 a un pouvoir de séparation négatif (-0,04) : il les juge
# plus proches que deux phrases synonymes. La variante multilingue obtient +0,60,
# pour la même dimension de sortie.
MODELE_NLP = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELE_CV = "openai/clip-vit-base-patch32"
DIM_NLP, DIM_CV = 384, 512

# Le téléchargement des images (phase 1) a nommé le dossier du compte perso "OWN"
DOSSIER_MEDIA = {"echoes.by.wiam": "OWN"}

# Tranches de hashtags : l'effet mesuré en phase 2 est en cloche, pas linéaire
TRANCHES_HASHTAGS = [-1, 0, 5, 10, 20, 10_000]
NOMS_HASHTAGS = ["0", "1-5", "6-10", "11-20", "21+"]

THEMES = ["art", "ecriture", "melange", "nature", "photographie"]
TYPES_POST = ["image", "carrousel", "video"]

# Standardisées sur le train uniquement, puis appliquées au reste
A_STANDARDISER = ["followers", "jours_depuis_dernier_post"]


def chemin_media(ligne):
    dossier = DOSSIER_MEDIA.get(ligne.compte, ligne.compte)
    return MEDIA / dossier / f"{ligne.id}.jpg"


def normaliser_l2(vecteurs):
    """Normalisation par ligne : chaque vecteur indépendamment, donc aucune fuite."""
    normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)
    return np.divide(vecteurs, normes, out=np.zeros_like(vecteurs), where=normes > 0)


def charger_cache(chemin_vecs, chemin_ids, dim):
    if not (chemin_vecs.exists() and chemin_ids.exists()):
        return {}
    vecs, ids = np.load(chemin_vecs), np.load(chemin_ids)
    if vecs.shape[0] != ids.shape[0] or vecs.shape[1] != dim:
        return {}
    return dict(zip(ids.tolist(), vecs))


def sauver_cache(cache, chemin_vecs, chemin_ids):
    ids = np.array(list(cache.keys()), dtype=np.int64)
    vecs = np.stack([cache[i] for i in ids.tolist()]).astype(np.float32)
    np.save(chemin_vecs, vecs)
    np.save(chemin_ids, ids)


def embeddings_nlp(df, recalc):
    cache = {} if recalc else charger_cache(
        PROC / "embeddings_nlp.npy", PROC / "embeddings_nlp_ids.npy", DIM_NLP)

    legendes = df.legende.fillna("").str.strip()
    a_calculer = [(i, t) for i, t in zip(df.id, legendes)
                  if t and i not in cache]

    if a_calculer:
        from sentence_transformers import SentenceTransformer
        print(f"  {len(a_calculer)} legendes a encoder ({MODELE_NLP})")
        modele = SentenceTransformer(MODELE_NLP)
        vecs = modele.encode([t for _, t in a_calculer], batch_size=64,
                             show_progress_bar=True, convert_to_numpy=True)
        cache.update({i: v for (i, _), v in zip(a_calculer, vecs)})
        sauver_cache(cache, PROC / "embeddings_nlp.npy", PROC / "embeddings_nlp_ids.npy")
    else:
        print("  cache NLP complet, rien a recalculer")

    zero = np.zeros(DIM_NLP, dtype=np.float32)
    matrice = np.stack([cache.get(i, zero) for i in df.id])
    vide = legendes.eq("").to_numpy()
    matrice[vide] = 0.0
    return normaliser_l2(matrice), vide


def embeddings_cv(df, recalc):
    cache = {} if recalc else charger_cache(
        PROC / "embeddings_cv.npy", PROC / "embeddings_cv_ids.npy", DIM_CV)

    chemins = {r.id: chemin_media(r) for r in df.itertuples()}
    manquante = np.array([not chemins[i].exists() for i in df.id])
    a_calculer = [i for i in df.id if chemins[i].exists() and i not in cache]

    if a_calculer:
        import torch
        from PIL import Image
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        print(f"  {len(a_calculer)} images a encoder ({MODELE_CV})")
        processeur = CLIPImageProcessor.from_pretrained(MODELE_CV)
        modele = CLIPVisionModelWithProjection.from_pretrained(MODELE_CV).eval()

        lot = 32
        for debut in range(0, len(a_calculer), lot):
            ids = a_calculer[debut:debut + lot]
            images = [Image.open(chemins[i]).convert("RGB") for i in ids]
            entrees = processeur(images=images, return_tensors="pt")
            with torch.no_grad():
                sorties = modele(**entrees).image_embeds.numpy().astype(np.float32)
            cache.update(dict(zip(ids, sorties)))
            print(f"    {min(debut + lot, len(a_calculer))}/{len(a_calculer)}", end="\r")
        print()
        sauver_cache(cache, PROC / "embeddings_cv.npy", PROC / "embeddings_cv_ids.npy")
    else:
        print("  cache CV complet, rien a recalculer")

    zero = np.zeros(DIM_CV, dtype=np.float32)
    matrice = np.stack([cache.get(i, zero) for i in df.id])
    return normaliser_l2(matrice), manquante


def features_tabulaires(df):
    f = pd.DataFrame({"id": df.id.values})

    # --- temporel : encodage cyclique, pour que 23 h soit voisin de 0 h
    angle = 2 * np.pi * df.heure / 24
    f["heure_sin"] = np.sin(angle).values
    f["heure_cos"] = np.cos(angle).values
    # jour_semaine volontairement absent : aucun effet mesure en phase 2 (p = 0,86)

    ordre = df.sort_values(["compte", "timestamp"])
    gap = ordre.groupby("compte").timestamp.diff().dt.total_seconds() / 86400
    f["jours_depuis_dernier_post"] = gap.reindex(df.index).values

    # --- texte : seul le comptage de hashtags est retenu, en tranches
    f["hashtags_tranche"] = pd.cut(df.nb_hashtags, TRANCHES_HASHTAGS,
                                   labels=range(len(NOMS_HASHTAGS))).astype(int).values

    # --- compte et format
    f["followers"] = df.followers.values
    for t in TYPES_POST:
        f[f"type_{t}"] = df.type_post.eq(t).astype(int).values
    for t in THEMES:
        f[f"theme_{t}"] = df.theme.eq(t).astype(int).values
    return f


def standardiser(f, masque_train):
    """Moyenne et écart-type calculés sur le train seul, puis appliqués partout."""
    params = {}
    for col in A_STANDARDISER:
        ref = f.loc[masque_train, col]
        moyenne, ecart = float(ref.mean()), float(ref.std())
        ecart = ecart if ecart > 0 else 1.0
        f[f"{col}_z"] = (f[col] - moyenne) / ecart
        params[col] = {"moyenne": moyenne, "ecart_type": ecart,
                       "n_train": int(ref.notna().sum())}
    PARAMS.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return f, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recalc", action="store_true", help="ignore le cache d'embeddings")
    args = ap.parse_args()

    df = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    print(f"{len(df)} posts charges")

    print("NLP :")
    vec_nlp, caption_vide = embeddings_nlp(df, args.recalc)
    print("CV :")
    vec_cv, image_manquante = embeddings_cv(df, args.recalc)

    f = features_tabulaires(df)
    f["caption_vide"] = caption_vide.astype(int)
    f["image_manquante"] = image_manquante.astype(int)

    # Cible : uniquement la ou la valeur est connue et le post entrainable
    f["log_interactions"] = np.where(df.utilisable_entrainement,
                                     np.log1p(df.interactions), np.nan)
    f["groupe_split"] = df.groupe_split.values
    f["utilisable_entrainement"] = df.utilisable_entrainement.values

    masque_train = (df.groupe_split == "train") & df.utilisable_entrainement
    f, params = standardiser(f, masque_train.values)

    # Les embeddings sont normalisés dans [-1, 1] : six décimales suffisent largement
    # et divisent la taille du fichier par deux. La cible, elle, garde sa précision.
    nlp = pd.DataFrame(vec_nlp.round(6), columns=[f"nlp_{k:03d}" for k in range(DIM_NLP)])
    cv = pd.DataFrame(vec_cv.round(6), columns=[f"cv_{k:03d}" for k in range(DIM_CV)])
    complet = pd.concat([f.reset_index(drop=True), nlp, cv], axis=1)

    complet.to_csv(SORTIE, index=False)

    print(f"\n{SORTIE} : {complet.shape[0]} lignes x {complet.shape[1]} colonnes"
          f" ({SORTIE.stat().st_size // 1_048_576} Mo)")
    print(f"  cible renseignee   : {complet.log_interactions.notna().sum()}")
    print(f"  legendes vides     : {int(f.caption_vide.sum())}")
    print(f"  images manquantes  : {int(f.image_manquante.sum())}")
    print(f"  premier post/compte sans ecart : {int(f.jours_depuis_dernier_post.isna().sum())}")
    print(f"\nStandardisation calculee sur {params['followers']['n_train']} posts de train :")
    for col, p in params.items():
        print(f"  {col:28} moyenne={p['moyenne']:.2f} ecart-type={p['ecart_type']:.2f}")


if __name__ == "__main__":
    main()
