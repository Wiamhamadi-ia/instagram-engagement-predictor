"""Fusionne les données brutes en un dataset propre : data/processed/dataset_clean.csv

Usage:
    python -m src.processing.clean_data
"""
import json
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed/dataset_clean.csv")

TYPE_POST = {"IMAGE": "image", "CAROUSEL_ALBUM": "carrousel", "VIDEO": "video"}
OWN_ACCOUNT = "echoes.by.wiam"
VAL_RATIO = 0.25

# Plages Unicode des emojis (pictogrammes, symboles, drapeaux, dingbats)
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF☀-➿⬀-⯿️←-⇿]"
)
HASHTAG = re.compile(r"#\w+", re.UNICODE)
MENTION = re.compile(r"@\w+")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame(json.loads(line) for line in f)


def split_par_compte(comptes_tries, val_ratio=VAL_RATIO):
    """Assigne chaque compte à train ou validation.

    Le découpage est fait par compte, pas par post : deux posts d'un même compte
    partagent son audience et son style, donc les séparer ferait fuiter de
    l'information du train vers la validation. Les comptes sont pris en
    alternance dans l'ordre décroissant de volume, pour que les deux groupes
    aient des tailles de comptes comparables.
    """
    attrib, n_val_cible = {}, round(len(comptes_tries) * val_ratio)
    n_val = 0
    for i, compte in enumerate(comptes_tries):
        pas = max(1, round(1 / val_ratio))
        if n_val < n_val_cible and i % pas == pas - 1:
            attrib[compte] = "validation"
            n_val += 1
        else:
            attrib[compte] = "train"
    return attrib


def main():
    posts = read_jsonl(RAW / "posts.jsonl")
    profils = read_jsonl(RAW / "profiles.jsonl")
    comptes = pd.read_csv(RAW / "accounts.csv")

    # OWN est le nom interne du compte perso ; profiles.jsonl le nomme par son username
    posts["compte"] = posts["account"].replace({"OWN": OWN_ACCOUNT})

    df = posts.merge(
        profils[["username", "followers_count", "follows_count", "media_count"]],
        left_on="compte", right_on="username", how="left",
    ).drop(columns=["username", "account"])

    df = df.merge(
        comptes[["username", "approx_followers"]].rename(
            columns={"approx_followers": "followers_annonces"}),
        left_on="compte", right_on="username", how="left",
    ).drop(columns=["username"])

    df = df.rename(columns={
        "theme": "theme", "like_count": "likes", "comments_count": "commentaires",
        "followers_count": "followers", "media_count": "posts_publies_total",
        "follows_count": "abonnements", "caption": "legende",
    })

    df["type_post"] = df["media_type"].map(TYPE_POST)
    df["est_reel"] = df["media_product_type"].eq("REELS")
    df["likes_masques"] = df["likes"].isna()

    # L'API masque les likes sur certains comptes, mais jamais les commentaires.
    df["interactions"] = df["likes"].fillna(0) + df["commentaires"]
    df["taux_engagement"] = (df["interactions"] / df["followers"] * 100).where(
        ~df["likes_masques"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
    df["annee"] = df["timestamp"].dt.year
    df["heure"] = df["timestamp"].dt.hour
    df["jour_semaine"] = df["timestamp"].dt.dayofweek
    df["nom_jour"] = df["timestamp"].dt.day_name()

    legende = df["legende"].fillna("")
    df["longueur_legende"] = legende.str.len()
    df["nb_mots_legende"] = legende.str.split().str.len().where(legende.ne(""), 0)
    df["nb_hashtags"] = legende.apply(lambda t: len(HASHTAG.findall(t)))
    df["nb_mentions"] = legende.apply(lambda t: len(MENTION.findall(t)))
    df["nb_emojis"] = legende.apply(lambda t: len(EMOJI.findall(t)))
    df["a_emoji"] = df["nb_emojis"] > 0
    df["a_legende"] = legende.ne("")

    # Le compte perso est le cas d'usage final, jamais un exemple d'entraînement
    volumes = df[df["compte"] != OWN_ACCOUNT]["compte"].value_counts()
    attrib = split_par_compte(list(volumes.index))
    df["groupe_split"] = df["compte"].map(attrib).fillna("cas_usage")

    df["utilisable_entrainement"] = (
        ~df["likes_masques"] & df["followers"].notna() & (df["groupe_split"] != "cas_usage")
    )

    colonnes = [
        "id", "compte", "theme", "permalink", "timestamp", "date", "annee", "heure",
        "jour_semaine", "nom_jour", "type_post", "est_reel", "legende",
        "longueur_legende", "nb_mots_legende", "nb_hashtags", "nb_mentions",
        "nb_emojis", "a_emoji", "a_legende", "likes", "commentaires", "interactions",
        "likes_masques", "followers", "followers_annonces", "abonnements",
        "posts_publies_total", "taux_engagement", "groupe_split",
        "utilisable_entrainement",
    ]
    df = df[colonnes].sort_values(["compte", "timestamp"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")

    print(f"{len(df)} posts -> {OUT}")
    print(f"  utilisables pour l'entrainement : {df['utilisable_entrainement'].sum()}")
    print(f"  likes masques                   : {df['likes_masques'].sum()}")
    print(df.groupby("groupe_split").agg(
        comptes=("compte", "nunique"), posts=("id", "count"),
        utilisables=("utilisable_entrainement", "sum")).to_string())


if __name__ == "__main__":
    main()
