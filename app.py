"""App Streamlit — estimer si un post fera mieux que l'ordinaire d'un compte.

Lancement :
    streamlit run app.py

Le modèle ne prédit pas un nombre de réactions, et l'interface ne prétend pas le
contraire : son AUC vaut 0,52, à peine au-dessus du hasard. Elle affiche donc une
tendance et sa fiabilité réelle, jamais un chiffre.
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parent
ARTEFACTS = RACINE / "models/app/artefacts.joblib"
REFERENCE = RACINE / "models/app/reference.json"

MODELE_NLP = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELE_CV = "openai/clip-vit-base-patch32"

HASHTAG = re.compile(r"#\w+", re.UNICODE)
TRANCHES = [-1, 0, 5, 10, 20, 10_000]
MES_ABONNES = 48   # valeur par défaut, modifiable

st.set_page_config(page_title="Prédire l'engagement d'un post",
                   page_icon="📊", layout="centered")


# --- chargement, une seule fois ------------------------------------------------

@st.cache_resource(show_spinner="Chargement du modèle de texte…")
def modele_texte():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODELE_NLP)


@st.cache_resource(show_spinner="Chargement du modèle d'image…")
def modele_image():
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    processeur = CLIPImageProcessor.from_pretrained(MODELE_CV)
    reseau = CLIPVisionModelWithProjection.from_pretrained(MODELE_CV).eval()
    return processeur, reseau


@st.cache_resource
def artefacts():
    return joblib.load(ARTEFACTS), json.loads(REFERENCE.read_text(encoding="utf-8"))


# --- pipeline de prédiction ----------------------------------------------------

def normaliser(v):
    norme = np.linalg.norm(v)
    return v / norme if norme > 0 else v


def encoder_legende(texte):
    if not texte.strip():
        return np.zeros(384, dtype=np.float32), True
    return normaliser(modele_texte().encode(texte, convert_to_numpy=True)), False


def encoder_image(fichier):
    import torch
    from PIL import Image
    processeur, reseau = modele_image()
    image = Image.open(fichier).convert("RGB")
    with torch.no_grad():
        vecteur = reseau(**processeur(images=[image], return_tensors="pt")).image_embeds
    return normaliser(vecteur.numpy()[0].astype(np.float32))


def predire(image, legende, heure, type_post, art):
    """Construit les mêmes variables qu'à l'entraînement, dans le même ordre.

    Le centrage se fait sur la moyenne du jeu d'entraînement : un nouvel
    utilisateur n'a pas d'historique d'embeddings à sa disposition. C'est une
    approximation assumée, et elle est mesurée — AUC 0,524 contre 0,530 avec le
    vrai centrage par compte.
    """
    vec_nlp, vide = encoder_legende(legende)
    vec_cv = encoder_image(image)

    angle = 2 * np.pi * heure / 24
    nb_hashtags = len(HASHTAG.findall(legende))
    tranche = int(pd.cut([nb_hashtags], TRANCHES, labels=range(5))[0])

    tabulaire = {
        "heure_sin": np.sin(angle), "heure_cos": np.cos(angle),
        "jours_depuis_dernier_post": art["mediane_gap"],
        "hashtags_tranche": tranche,
        "type_image": int(type_post == "image"),
        "type_carrousel": int(type_post == "carrousel"),
        "type_video": int(type_post == "vidéo"),
        "caption_vide": int(vide), "image_manquante": 0,
    }
    tab = np.array([[tabulaire[c] for c in art["colonnes_tabulaires"]]])

    X = np.hstack([
        tab,
        art["acp_nlp"].transform((vec_nlp - art["moyenne_nlp"]).reshape(1, -1)),
        art["acp_cv"].transform((vec_cv - art["moyenne_cv"]).reshape(1, -1)),
    ])
    return float(art["modele"].predict(X)[0]), nb_hashtags


def tendance(score, seuils):
    if score < seuils[0]:
        return "en dessous", "🔽", ("Ce post se situerait plutôt **en dessous** de "
                                    "l'ordinaire de ce compte.")
    if score > seuils[1]:
        return "au-dessus", "🔼", ("Ce post se situerait plutôt **au-dessus** de "
                                   "l'ordinaire de ce compte.")
    return "dans la moyenne", "➖", ("Ce post ressemble à ce que ce compte fait "
                                     "**d'habitude**.")


def facteurs(heure, nb_hashtags, ref):
    """Deux repères issus de la phase 2, descriptifs et mesurés."""
    notes = []
    pic = ref["phase2"]["heure_pic"]
    ecart = min(abs(heure - pic), 24 - abs(heure - pic))
    if ecart <= 2:
        notes.append(f"🕘 **Heure favorable** — {heure} h UTC est proche du pic "
                     f"observé à {pic} h, qui apporte environ +25 % dans les données.")
    elif ecart >= 8:
        notes.append(f"🕘 **Heure peu favorable** — {heure} h UTC est loin du pic "
                     f"observé à {pic} h.")
    else:
        notes.append(f"🕘 **Heure neutre** — {heure} h UTC n'est ni le meilleur ni le "
                     f"pire créneau ({pic} h est le pic observé).")

    if 1 <= nb_hashtags <= 5:
        notes.append(f"#️⃣ **Hashtags bien dosés** — {nb_hashtags} hashtags, dans la "
                     f"zone qui fait +9 % dans les données.")
    elif nb_hashtags == 0:
        notes.append("#️⃣ **Aucun hashtag** — les posts sans hashtag font −8 % dans "
                     "les données ; un à cinq suffisent.")
    else:
        notes.append(f"#️⃣ **Beaucoup de hashtags** — {nb_hashtags} hashtags. "
                     f"Au-delà de cinq, le bénéfice disparaît dans les données.")
    return notes


# --- pages ---------------------------------------------------------------------

def page_prediction(art, ref):
    st.title("Ce post fera-t-il mieux que d'habitude ?")
    st.markdown(
        "Cet outil n'estime **pas** un nombre de mentions J'aime. Il indique si un "
        "post semble se situer au-dessus ou en dessous de ce que le compte obtient "
        "d'ordinaire — et il annonce à quel point cette indication est fiable."
    )

    gauche, droite = st.columns([1, 1])
    with gauche:
        image = st.file_uploader("Image du post", type=["jpg", "jpeg", "png", "webp"])
        if image:
            st.image(image, width="stretch")
    with droite:
        legende = st.text_area("Légende", height=180,
                               placeholder="Le texte du post, hashtags compris…")
        type_post = st.radio("Format", ["image", "carrousel", "vidéo"], horizontal=True,
                             help="Pour une vidéo, envoyez la miniature.")
        heure = st.slider("Heure de publication prévue (UTC)", 0, 23, 9)

    with st.expander("Contexte du compte (améliore la lecture du résultat)"):
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Abonnés du compte", min_value=1, value=MES_ABONNES,
                            step=1, key="abonnes",
                            help="Sert au repérage, pas à la prédiction : le modèle "
                                 "raisonne en écart au niveau habituel du compte.")
        with col2:
            niveau = st.number_input(
                "Niveau habituel (réactions médianes par post)", min_value=0,
                value=0, step=1,
                help="Laissez 0 si vous ne le connaissez pas.")
        recentes = st.text_input(
            "Je ne le connais pas — réactions de mes derniers posts",
            placeholder="ex : 42, 31, 58, 27, 44",
            help="Mentions J'aime + commentaires, séparés par des virgules.")
        if recentes.strip():
            valeurs = [float(v) for v in re.findall(r"\d+(?:[.,]\d+)?",
                                                    recentes.replace(",", " "))]
            if valeurs:
                niveau = float(np.median(valeurs))
                st.success(f"Niveau habituel estimé sur {len(valeurs)} posts : "
                           f"**{niveau:.0f} réactions** par post.")

    if not image:
        st.info("Ajoutez une image pour lancer l'analyse.")
        return

    if st.button("Analyser ce post", type="primary", width="stretch"):
        with st.spinner("Analyse de l'image et du texte…"):
            score, nb_hashtags = predire(image, legende, heure, type_post, art)

        libelle, icone, phrase = tendance(score, ref["seuils_tiers"])
        st.divider()
        st.subheader(f"{icone} {libelle.capitalize()}")
        st.markdown(phrase)

        if niveau > 0:
            bas, haut = niveau * np.exp(-1.12), niveau * np.exp(1.12)
            st.markdown(
                f"Pour situer : votre niveau habituel est de **{niveau:.0f} réactions**. "
                f"Dans les données, un post d'un même compte varie typiquement entre "
                f"**{bas:.0f}** et **{haut:.0f}** — c'est l'ampleur du hasard, bien "
                f"plus large que ce que le modèle sait anticiper."
            )

        st.divider()
        st.markdown("#### Ce qui joue dans cette estimation")
        for note in facteurs(heure, nb_hashtags, ref):
            st.markdown(note)

        st.divider()
        auc = ref["auc_conditions_app"]
        st.markdown("#### Quelle confiance accorder à ce résultat ?")
        st.progress((auc - 0.5) / 0.5,
                    text=f"Fiabilité mesurée : AUC {auc:.3f} sur 0,500 au hasard")
        st.warning(
            f"**Cette indication reflète une tendance faible, pas une certitude.** "
            f"Face à deux posts dont l'un a fait mieux que l'habitude et l'autre "
            f"moins bien, le modèle désigne le bon dans **{auc * 100:.0f} % des cas**, "
            f"contre 50 % en tirant à pile ou face. L'essentiel de la variation "
            f"d'engagement échappe au contenu du post."
        )


def page_limites(ref):
    st.title("Comprendre les limites")
    e = ref["entrainement"]

    st.markdown(f"""
Ce projet a produit un résultat modeste, et cette page existe pour ne pas le masquer.

#### Le modèle capte un signal faible

Son AUC vaut **{ref['auc_conditions_app']:.3f}**, contre 0,500 pour un tirage au sort.
Autrement dit, il a raison dans **{ref['auc_conditions_app'] * 100:.0f} % des cas** là
où le hasard en donne 50. C'est mesurable, mais très loin d'une certitude.

Mesuré autrement : les modèles testés **ne battent jamais** le pari trivial
« ce post fera comme d'habitude » sur l'erreur de prédiction. C'est pourquoi aucun
nombre de réactions n'est affiché.

#### L'essentiel de la variation échappe au contenu

L'écart-type de la cible vaut {e['ecart_type_cible']:.2f}, soit un **facteur 3** entre
deux posts d'un même compte. Une large part tient à la diffusion algorithmique
d'Instagram, que rien dans une image ou une légende ne permet d'anticiper.

#### Les vidéos ne sont vues que par leur miniature

**{e['part_video']:.0%} du jeu d'entraînement** est constitué de reels, dont seule
l'image de couverture est analysée. Le mouvement, le rythme, le montage et le son —
c'est-à-dire l'essentiel de ce qui fait marcher un reel — sont totalement absents du
modèle. Pour un carrousel, seule la première image compte.

#### L'échantillon est étroit

Le modèle a appris sur **{e['posts']} publications de {e['comptes']} comptes**, dont
l'audience va de **{e['followers_min']} à {e['followers_max']} abonnés**, tous dans des
univers créatifs : art, photographie, écriture, nature.

Un compte nettement en dehors de cette fourchette reçoit des estimations moins fiables
encore. C'est le cas du compte de référence de ce projet, avec ses 48 abonnés : il se
situe **sous le plancher** du jeu d'entraînement.

#### Une approximation propre à l'application

À l'entraînement, chaque image et chaque légende étaient comparées au style habituel
de leur compte. Un nouvel utilisateur n'ayant pas cet historique, l'application
compare au style moyen de l'ensemble du jeu. Le coût est mesuré : l'AUC passe de
{ref['auc_centrage_par_compte']:.3f} à {ref['auc_conditions_app']:.3f}.
""")

    st.info(
        "**Ce qui reste solide, ce sont les constats descriptifs**, issus de "
        "l'analyse de 3 587 publications. Ils figurent dans l'onglet *Explorer les "
        "résultats* et ne dépendent d'aucun modèle prédictif."
    )


def page_exploration(ref):
    import matplotlib.pyplot as plt
    from src.viz.style import DIVERGENT, INK, appliquer_style, couleurs_theme, finir

    ink = appliquer_style("light")
    couleurs, div = couleurs_theme("light"), DIVERGENT["light"]
    p2 = ref["phase2"]

    st.title("Ce que disent les données")
    st.markdown(
        "Ces constats viennent de l'analyse descriptive de **3 587 publications** de "
        "28 comptes. Contrairement à la prédiction, ils reposent sur des écarts "
        "mesurés et testés, pas sur un modèle. Chaque post est comparé à la médiane "
        "de son propre compte, pour que la notoriété ne fausse pas la lecture."
    )

    st.subheader("L'heure de publication compte, modérément")
    heures = np.arange(24)
    rel = np.array(p2["heure"]["relatif"])
    effectifs = np.array(p2["heure"]["n"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(heures, rel, color=couleurs["art"], marker="o", markersize=7,
            markeredgecolor=ink["surface"], markeredgewidth=2)
    ax.axhline(1.0, color=ink["primary"], linestyle="--", linewidth=1.5)
    maigres = effectifs < 60
    ax.scatter(heures[maigres], rel[maigres], s=150, facecolor="none",
               edgecolor=ink["muted"], linewidth=2, zorder=4)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h} h" for h in range(0, 24, 3)])
    ax.set_yticks([0.8, 1.0, 1.2])
    ax.set_yticklabels(["−20 %", "niveau habituel", "+20 %"])
    ax.margins(y=0.2)
    finir(ax, "Engagement selon l'heure de publication (UTC)",
          "cercles gris = créneaux trop peu fournis pour conclure", None, None, "light")
    st.pyplot(fig, width="stretch")
    st.caption(f"Le pic se situe à {p2['heure_pic']} h UTC, avec environ +25 % par "
               f"rapport à l'ordinaire du compte. L'écart entre le meilleur et le pire "
               f"créneau est d'un facteur 1,7 — réel, mais loin des promesses "
               f"habituelles.")

    st.subheader("Les hashtags : utiles en petit nombre, inutiles en pagaille")
    ecarts = np.array(p2["hashtags"]["relatif"]) - 1
    fig, ax = plt.subplots(figsize=(8, 3.4))
    barres = ax.bar(range(len(ecarts)), ecarts, width=0.6,
                    color=[div["pos"] if v >= 0 else div["neg"] for v in ecarts],
                    edgecolor=ink["surface"], linewidth=2)
    ax.axhline(0, color=ink["primary"], linewidth=1.2)
    for barre, v in zip(barres, ecarts):
        ax.annotate(f"{v * 100:+.0f} %", (barre.get_x() + barre.get_width() / 2, v),
                    xytext=(0, 4 if v >= 0 else -15), textcoords="offset points",
                    ha="center", fontsize=10, color=ink["primary"], fontweight="semibold")
    ax.set_xticks(range(len(ecarts)))
    ax.set_xticklabels([f"{l}\n{n} posts" for l, n in
                        zip(p2["hashtags"]["libelles"], p2["hashtags"]["n"])])
    ax.set_yticks([-0.05, 0, 0.05, 0.10])
    ax.set_yticklabels(["−5 %", "niveau habituel", "+5 %", "+10 %"])
    ax.margins(y=0.3)
    ax.grid(axis="x", visible=False)
    finir(ax, "Effet du nombre de hashtags", None, None, None, "light")
    st.pyplot(fig, width="stretch")
    st.caption("Un à cinq hashtags font +9 % par rapport à l'ordinaire du compte. "
               "Au-delà de cinq, le bénéfice disparaît entièrement.")

    st.subheader("Le jour de la semaine ne change rien")
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    ecarts_j = np.array(p2["jour"]["relatif"]) - 1
    fig, ax = plt.subplots(figsize=(8, 3.2))
    barres = ax.bar(range(7), ecarts_j, width=0.6,
                    color=[div["pos"] if v >= 0 else div["neg"] for v in ecarts_j],
                    edgecolor=ink["surface"], linewidth=2)
    ax.axhline(0, color=ink["primary"], linewidth=1.2)
    for barre, v in zip(barres, ecarts_j):
        ax.annotate(f"{v * 100:+.1f} %", (barre.get_x() + barre.get_width() / 2, v),
                    xytext=(0, 4 if v >= 0 else -15), textcoords="offset points",
                    ha="center", fontsize=9, color=ink["primary"])
    ax.set_xticks(range(7)); ax.set_xticklabels(jours)
    ax.set_yticks([-0.05, 0, 0.05])
    ax.set_yticklabels(["−5 %", "niveau habituel", "+5 %"])
    ax.margins(y=0.35)
    ax.grid(axis="x", visible=False)
    finir(ax, "Effet du jour de publication", None, None, None, "light")
    st.pyplot(fig, width="stretch")
    st.caption("Les sept jours tiennent dans une bande de ±4 %, ce qui est du bruit à "
               "ce volume. Soigner le créneau horaire a du sens, choisir le jour non.")


def main():
    if not ARTEFACTS.exists():
        st.error("Artefacts absents. Lancez d'abord :\n\n"
                 "`python -m src.models.export_app`")
        return

    art, ref = artefacts()
    st.sidebar.title("Engagement Instagram")
    st.sidebar.caption("Projet de data science — phases 1 à 5")
    page = st.sidebar.radio("Navigation", ["Prédire un post",
                                           "Explorer les résultats",
                                           "Comprendre les limites"])
    st.sidebar.divider()
    st.sidebar.metric("Fiabilité du modèle", f"AUC {ref['auc_conditions_app']:.3f}",
                      delta=f"{ref['auc_conditions_app'] - 0.5:+.3f} vs hasard",
                      help="0,500 correspond à un tirage au sort.")
    st.sidebar.caption(f"Entraîné sur {ref['entrainement']['posts']} publications "
                       f"de {ref['entrainement']['comptes']} comptes.")

    if page == "Prédire un post":
        page_prediction(art, ref)
    elif page == "Explorer les résultats":
        page_exploration(ref)
    else:
        page_limites(ref)


if __name__ == "__main__":
    main()
