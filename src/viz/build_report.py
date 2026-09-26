"""Génère reports/exploration_summary.html — résumé visuel destiné au portfolio.

Lecteur visé : non technique. Pas de code, pas de jargon statistique dans le texte
visible. Chaque graphique est rendu en clair et en sombre, la page bascule selon le
thème du lecteur.

Usage:
    python -m src.viz.build_report
"""
import base64
import io
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz.style import (DIVERGENT, INK, THEMES, TYPES_POST, appliquer_style,
                           couleurs_theme, couleurs_type)

SOURCE = Path("data/processed/dataset_clean.csv")
SORTIE = Path("reports/exploration_summary.html")
JOURS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
JOURS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def en_svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue().encode("utf-8")).decode("ascii")


def titre_axe(ax, texte, ink):
    ax.set_title(texte, color=ink["primary"], fontsize=11, pad=10)


# --- graphiques ---------------------------------------------------------------

def fig_distribution(u, mode):
    ink, c = INK[mode], couleurs_theme(mode)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    pos = u.taux_engagement[u.taux_engagement > 0]
    bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 45)
    ax.hist(pos, bins=bins, color=c["art"], edgecolor="none")
    ax.set_xscale("log")
    med = u.taux_engagement.median()
    ax.axvline(med, color=ink["primary"], linestyle="--", linewidth=1.5)
    ax.annotate(f"moitié des posts\nsous {med:.1f} %", (med, ax.get_ylim()[1] * 0.95),
                xytext=(10, 0), textcoords="offset points", va="top",
                fontsize=9, color=ink["primary"], fontweight="semibold")
    ax.set_xlabel("engagement du post (%)", color=ink["secondary"])
    ax.set_ylabel("nombre de posts", color=ink["secondary"])
    ax.grid(axis="x", visible=False)
    return en_svg(fig)


def fig_heure(u, mode):
    ink, c = INK[mode], couleurs_theme(mode)
    h = u.groupby("heure").taux_engagement.median()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(h.index, h.values, color=c["art"], marker="o", markersize=8,
            markeredgecolor=ink["surface"], markeredgewidth=2, solid_capstyle="round")
    pic, creux = h.idxmax(), h.idxmin()
    ax.annotate(f"{h.max():.1f} % à {pic} h", (pic, h.max()), xytext=(0, 12),
                textcoords="offset points", ha="center", fontsize=10,
                color=ink["primary"], fontweight="semibold")
    ax.annotate(f"{h.min():.1f} % à {creux} h", (creux, h.min()), xytext=(0, -20),
                textcoords="offset points", ha="center", fontsize=10,
                color=ink["secondary"])
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{k} h" for k in range(0, 24, 3)])
    ax.margins(y=0.22)
    ax.set_ylabel("engagement typique (%)", color=ink["secondary"])
    ax.grid(axis="x", visible=False)
    return en_svg(fig)


def fig_type_post(u, mode):
    """Deux panneaux : le cas courant, puis la moyenne que les viraux tirent."""
    ink, c = INK[mode], couleurs_type(mode)
    stat = u.groupby("type_post").taux_engagement.agg(["median", "mean"]).reindex(TYPES_POST)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, col, titre in [(axes[0], "median", "Le post habituel"),
                           (axes[1], "mean", "La moyenne, viraux compris")]:
        vals = stat[col].values
        b = ax.bar(range(3), vals, width=0.6, color=[c[t] for t in TYPES_POST],
                   edgecolor=ink["surface"], linewidth=2)
        for barre, v in zip(b, vals):
            ax.annotate(f"{v:.1f} %", (barre.get_x() + barre.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=10, color=ink["primary"], fontweight="semibold")
        ax.set_xticks(range(3)); ax.set_xticklabels(TYPES_POST)
        ax.set_ylim(top=max(vals) * 1.25)
        titre_axe(ax, titre, ink)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("engagement (%)", color=ink["secondary"])
    return en_svg(fig)


def fig_hashtags(u, mode):
    """Écart au niveau habituel du compte, ancré à zéro (pas d'axe tronqué)."""
    ink, div = INK[mode], DIVERGENT[mode]
    noms = ["aucun", "1 à 5", "6 à 10", "11 à 20", "21 et +"]
    tr = pd.cut(u.nb_hashtags, [-1, 0, 5, 10, 20, 1000], labels=noms)
    rel = u.taux_engagement / u.groupby("compte").taux_engagement.transform("median")
    g = rel.groupby(tr, observed=True).median().reindex(noms) - 1.0
    n = u.groupby(tr, observed=True).size().reindex(noms)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    b = ax.bar(range(len(noms)), g.values, width=0.6,
               color=[div["pos"] if v >= 0 else div["neg"] for v in g.values],
               edgecolor=ink["surface"], linewidth=2)
    ax.axhline(0, color=ink["primary"], linewidth=1.2)
    for barre, v in zip(b, g.values):
        ax.annotate(f"{v * 100:+.0f} %", (barre.get_x() + barre.get_width() / 2, v),
                    xytext=(0, 4 if v >= 0 else -15), textcoords="offset points",
                    ha="center", fontsize=10, color=ink["primary"], fontweight="semibold")
    ax.set_xticks(range(len(noms)))
    ax.set_xticklabels([f"{t}\n{int(k)} posts" for t, k in zip(noms, n)])
    ax.set_yticks([-0.05, 0, 0.05, 0.10])
    ax.set_yticklabels(["−5 %", "niveau habituel", "+5 %", "+10 %"])
    ax.margins(y=0.3)
    ax.set_xlabel("nombre de hashtags", color=ink["secondary"])
    ax.grid(axis="x", visible=False)
    return en_svg(fig)


def fig_jour(u, mode):
    ink, c = INK[mode], couleurs_theme(mode)
    j = u.groupby("nom_jour").taux_engagement.median().reindex(JOURS_EN)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    b = ax.bar(range(7), j.values, width=0.62, color=c["art"],
               edgecolor=ink["surface"], linewidth=2)
    for barre, v in zip(b, j.values):
        ax.annotate(f"{v:.1f} %", (barre.get_x() + barre.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=10, color=ink["primary"], fontweight="semibold")
    ax.axhline(u.taux_engagement.median(), color=ink["muted"], linestyle="--", linewidth=1.2)
    ax.set_xticks(range(7)); ax.set_xticklabels(JOURS_FR)
    ax.set_ylim(0, j.max() * 1.45)
    ax.set_ylabel("engagement typique (%)", color=ink["secondary"])
    ax.grid(axis="x", visible=False)
    return en_svg(fig)


def fig_taille_compte(u, mode):
    ink, c = INK[mode], couleurs_theme(mode)
    parc = u.groupby("compte").agg(followers=("followers", "first"),
                                   med=("taux_engagement", "median"))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.scatter(parc.followers, parc.med, s=95, color=c["art"],
               edgecolor=ink["surface"], linewidth=2, zorder=3)
    # Tendance lissée sur les rangs, pour donner le sens de lecture
    ordre = parc.sort_values("followers")
    lisse = ordre.med.rolling(7, center=True, min_periods=3).median()
    ax.plot(ordre.followers, lisse, color=ink["muted"], linewidth=2, linestyle="--", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(parc.followers.min() * 0.75, parc.followers.max() * 1.5)
    ax.set_xlabel("abonnés du compte", color=ink["secondary"])
    ax.set_ylabel("engagement typique (%)", color=ink["secondary"])
    ax.annotate("tendance", (ordre.followers.iloc[-4], lisse.iloc[-4]),
                xytext=(6, 10), textcoords="offset points",
                fontsize=9, color=ink["secondary"])
    ax.grid(axis="x", visible=False)
    return en_svg(fig)


def fig_themes(u, mode):
    ink, c = INK[mode], couleurs_theme(mode)
    med = u.groupby("theme").taux_engagement.median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    b = ax.barh(range(len(med)), med.values, height=0.66,
                color=[c[t] for t in med.index], edgecolor=ink["surface"], linewidth=2)
    for i, v in enumerate(med.values):
        ax.annotate(f"{v:.1f} %", (v, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, color=ink["primary"], fontweight="semibold")
    ax.set_yticks(range(len(med))); ax.set_yticklabels(med.index)
    ax.invert_yaxis()
    ax.set_xlim(0, med.max() * 1.22)
    ax.set_xlabel("engagement typique (%)", color=ink["secondary"])
    ax.grid(axis="y", visible=False)
    return en_svg(fig)


FIGURES = [
    ("distribution", fig_distribution), ("heure", fig_heure),
    ("type_post", fig_type_post), ("hashtags", fig_hashtags), ("jour", fig_jour),
    ("taille", fig_taille_compte), ("themes", fig_themes),
]


def construire_figures(u):
    svgs = {}
    for mode in ("light", "dark"):
        appliquer_style(mode)
        for nom, fn in FIGURES:
            svgs[f"{nom}_{mode}"] = fn(u, mode)
    return svgs


def bloc_figure(svgs, nom, alt):
    return (f'<figure class="chart">'
            f'<img class="only-light" alt="{alt}" src="data:image/svg+xml;base64,{svgs[nom + "_light"]}">'
            f'<img class="only-dark" alt="{alt}" src="data:image/svg+xml;base64,{svgs[nom + "_dark"]}">'
            f"</figure>")


def main():
    df = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    u = df[df.utilisable_entrainement].copy()
    svgs = construire_figures(u)

    h = u.groupby("heure").taux_engagement.median()
    stat_type = u.groupby("type_post").taux_engagement.agg(["median", "mean"])
    med_theme = u.groupby("theme").taux_engagement.median().sort_values(ascending=False)
    kpi = {
        "posts": f"{len(u):,}".replace(",", " "),
        "comptes": u.compte.nunique(),
        "median": f"{u.taux_engagement.median():.1f} %",
        "viraux": f"{(u.taux_engagement > 100).mean() * 100:.1f} %",
        "ratio_heure": f"{h.max() / h.min():.1f}",
        "pic": f"{h.idxmax()} h", "creux": f"{h.idxmin()} h",
        "video_med": f"{stat_type.loc['video', 'median']:.1f} %",
        "video_moy": f"{stat_type.loc['video', 'mean']:.0f} %",
        "image_med": f"{stat_type.loc['image', 'median']:.1f} %",
        "image_moy": f"{stat_type.loc['image', 'mean']:.0f} %",
        "theme_haut": med_theme.index[0], "theme_haut_v": f"{med_theme.iloc[0]:.1f} %",
        "theme_bas": med_theme.index[-1], "theme_bas_v": f"{med_theme.iloc[-1]:.1f} %",
        "debut": u.timestamp.min().year, "fin": u.timestamp.max().year,
        "date": date.today().strftime("%d/%m/%Y"),
    }

    html = GABARIT.format(**kpi, **{f"fig_{n}": bloc_figure(svgs, n, a) for n, a in [
        ("distribution", "Répartition de l'engagement des posts"),
        ("heure", "Engagement selon l'heure de publication"),
        ("type_post", "Engagement par type de post"),
        ("hashtags", "Effet du nombre de hashtags sur l'engagement"),
        ("jour", "Engagement selon le jour de publication"),
        ("taille", "Engagement selon la taille du compte"),
        ("themes", "Engagement par thème de compte"),
    ]})
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(html, encoding="utf-8")
    print(f"{SORTIE} ({SORTIE.stat().st_size // 1024} Ko)")


GABARIT = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Ce qui fait réagir sur Instagram</title>
<style>
  :root {{
    color-scheme: light;
    --surface: #fcfcfb; --card: #ffffff; --border: #e4e3df;
    --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #8a8880;
    --accent: #2a78d6; --accent-soft: #eef4fd;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface: #1a1a19; --card: #232321; --border: #33332f;
      --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8a8880;
      --accent: #3987e5; --accent-soft: #1e2b3d;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface: #1a1a19; --card: #232321; --border: #33332f;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8a8880;
    --accent: #3987e5; --accent-soft: #1e2b3d;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--surface); color: var(--ink);
    font: 16px/1.65 "Segoe UI", system-ui, -apple-system, sans-serif;
    padding: 0 16px env(safe-area-inset-bottom, 0px);
  }}
  .wrap {{ max-width: 860px; margin: 0 auto; padding-block: 56px 72px; }}
  header {{ border-bottom: 1px solid var(--border); padding-bottom: 28px; margin-bottom: 40px; }}
  .kicker {{ color: var(--accent); font-size: 13px; font-weight: 600;
             letter-spacing: .08em; text-transform: uppercase; margin: 0 0 12px; }}
  h1 {{ font-size: clamp(28px, 5vw, 40px); line-height: 1.18; margin: 0 0 14px; letter-spacing: -.02em; }}
  .chapo {{ color: var(--ink-2); font-size: 17px; margin: 0; max-width: 62ch; }}
  .meta {{ color: var(--ink-3); font-size: 13px; margin-top: 18px; }}
  .kpis {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           margin: 0 0 48px; padding: 0; list-style: none; }}
  .kpis li {{ background: var(--card); border: 1px solid var(--border);
              border-radius: 12px; padding: 16px 18px; }}
  .kpis b {{ display: block; font-size: 26px; letter-spacing: -.02em; line-height: 1.2; }}
  .kpis span {{ color: var(--ink-2); font-size: 13px; }}
  section {{ margin-bottom: 52px; }}
  h2 {{ font-size: clamp(20px, 3.4vw, 25px); line-height: 1.3; margin: 0 0 10px; letter-spacing: -.01em; }}
  section > p {{ color: var(--ink-2); margin: 0 0 22px; max-width: 64ch; }}
  .chart {{ margin: 0 0 18px; }}
  .chart img {{ width: 100%; height: auto; display: block; }}
  .only-dark {{ display: none; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) .only-light {{ display: none; }}
    :root:not([data-theme="light"]) .only-dark {{ display: block; }}
  }}
  :root[data-theme="dark"] .only-light {{ display: none; }}
  :root[data-theme="dark"] .only-dark {{ display: block; }}
  .cle {{ background: var(--accent-soft); border-left: 3px solid var(--accent);
          border-radius: 0 10px 10px 0; padding: 14px 18px; color: var(--ink);
          font-size: 15px; }}
  .cle b {{ font-weight: 650; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 15px; margin-bottom: 18px; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--ink-2); font-weight: 600; font-size: 13px;
        text-transform: uppercase; letter-spacing: .05em; }}
  td.oui {{ color: var(--ink); font-weight: 600; }}
  td.non {{ color: var(--ink-3); }}
  .tablewrap {{ overflow-x: auto; }}
  footer {{ border-top: 1px solid var(--border); padding-top: 24px;
            color: var(--ink-3); font-size: 13px; }}
  footer p {{ margin: 0 0 8px; max-width: 70ch; }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <p class="kicker">Analyse de données · Instagram</p>
  <h1>Ce qui fait réagir sur Instagram, et ce qui n'y change rien</h1>
  <p class="chapo">
    {posts} publications de {comptes} comptes créatifs, publiées entre {debut} et {fin},
    passées au crible. Objectif : distinguer ce qui influence vraiment les réactions
    d'un post des conseils que tout le monde répète.
  </p>
  <p class="meta">Analyse réalisée le {date} · source : API Instagram Graph</p>
</header>

<ul class="kpis">
  <li><b>{posts}</b><span>publications analysées</span></li>
  <li><b>{comptes}</b><span>comptes suivis</span></li>
  <li><b>{median}</b><span>engagement d'un post typique</span></li>
  <li><b>{viraux}</b><span>de posts dépassant 100 %</span></li>
</ul>

<section>
  <h2>Un post ordinaire fait peu de bruit, quelques-uns explosent</h2>
  <p>
    Chaque barre compte les publications ayant obtenu un niveau de réaction donné.
    L'échelle horizontale est resserrée : chaque graduation représente dix fois plus
    que la précédente, sinon la quasi-totalité des posts s'entasserait à gauche.
  </p>
  {fig_distribution}
  <p class="cle">
    La moitié des publications reste sous <b>{median}</b> d'engagement, mais une
    minorité dépasse les 100 %, c'est-à-dire récolte plus de réactions que le compte
    n'a d'abonnés. Ces posts-là sont vus bien au-delà de la communauté du compte.
    <b>Parler de « moyenne » n'a donc aucun sens ici</b> : quelques succès la font
    exploser. Tous les chiffres de cette page décrivent le post typique.
  </p>
</section>

<section>
  <h2>L'heure de publication change presque tout</h2>
  <p>
    Engagement du post typique selon l'heure de mise en ligne, en temps universel (UTC).
  </p>
  {fig_heure}
  <p class="cle">
    Entre le pire et le meilleur créneau, l'écart est d'un facteur <b>{ratio_heure}</b> :
    de {creux} à {pic}, le même contenu ne reçoit pas du tout le même accueil.
    C'est le levier le plus fort de toute l'étude, et le plus simple à actionner.
  </p>
</section>

<section>
  <h2>La vidéo est un billet de loterie</h2>
  <p>
    À gauche, ce que rapporte une publication ordinaire. À droite, la moyenne, qui
    inclut les succès exceptionnels.
  </p>
  {fig_type_post}
  <p class="cle">
    Une vidéo ordinaire fait <b>moins bien qu'une simple photo</b> ({video_med} contre
    {image_med}). Mais en moyenne, elle rapporte quatre fois plus ({video_moy} contre
    {image_moy}), parce que les rares vidéos virales concentrent l'essentiel des
    réactions. La vidéo augmente les chances de carton tout en abaissant le résultat
    le plus probable.
  </p>
</section>

<section>
  <h2>Les hashtags : utiles en petit nombre, inutiles en pagaille</h2>
  <p>
    Chaque publication est comparée à ce que son propre compte obtient d'habitude.
    Cela évite de confondre l'effet des hashtags avec celui de la notoriété du compte :
    un gros compte reste comparé à lui-même.
  </p>
  {fig_hashtags}
  <p class="cle">
    Les publications portant <b>un à cinq hashtags dépassent de 9 %</b> le niveau
    habituel de leur compte. Au-delà, le bénéfice disparaît : à partir de six, et
    jusqu'à vingt et plus, les posts repassent légèrement sous ce niveau.
    <b>En empiler trente n'apporte rien.</b> À retenir tout de même : l'écart reste
    de l'ordre de 10 %, très loin du facteur {ratio_heure} de l'heure de publication.
  </p>
</section>

<section>
  <h2>Le jour de la semaine, lui, ne change rien</h2>
  <p>
    Engagement du post typique selon le jour de mise en ligne. Le trait pointillé
    marque le niveau habituel, toutes publications confondues.
  </p>
  {fig_jour}
  <p class="cle">
    Les sept barres tiennent dans un mouchoir de poche, autour du niveau de référence.
    Un mardi vaut un samedi. <b>L'heure compte, le jour non</b> — c'est le créneau
    horaire qu'il faut soigner, pas le choix du jour.
  </p>
</section>

<section>
  <h2>Les petits comptes font réagir davantage</h2>
  <p>Chaque point représente un compte : son nombre d'abonnés et son engagement habituel.</p>
  {fig_taille}
  <p class="cle">
    Plus un compte grandit, plus la part de sa communauté qui réagit diminue.
    Un petit compte très actif peut donc afficher un taux bien supérieur à celui
    d'un compte dix fois plus gros. <b>Comparer deux comptes de tailles différentes
    sur ce seul chiffre n'a pas de sens.</b>
  </p>
</section>

<section>
  <h2>Le sujet du compte pèse, mais moins qu'on ne croit</h2>
  <p>Engagement du post typique selon la thématique dominante du compte.</p>
  {fig_themes}
  <p class="cle">
    L'écart va de <b>{theme_bas_v}</b> ({theme_bas}) à <b>{theme_haut_v}</b>
    ({theme_haut}), soit un rapport de trois. Une partie de cet écart vient
    cependant du point précédent : les thématiques les mieux placées sont aussi
    celles qui rassemblent le plus de petits comptes.
  </p>
</section>

<section>
  <h2>Ce qui compte, ce qui ne compte pas</h2>
  <div class="tablewrap">
  <table>
    <thead><tr><th>Facteur</th><th>Influence mesurée</th></tr></thead>
    <tbody>
      <tr><td class="oui">Heure de publication</td><td class="oui">Forte — facteur {ratio_heure}</td></tr>
      <tr><td class="oui">Taille du compte</td><td class="oui">Forte — en sens inverse</td></tr>
      <tr><td class="oui">Thème du compte</td><td class="oui">Modérée — rapport de 3</td></tr>
      <tr><td class="oui">Format du post</td><td class="oui">Modérée — photo devant vidéo</td></tr>
      <tr><td class="oui">Hashtags (1 à 5)</td><td class="oui">Faible — environ +9 %</td></tr>
      <tr><td class="non">Longueur de la légende</td><td class="non">Très faible</td></tr>
      <tr><td class="non">Présence d'emojis</td><td class="non">Aucune</td></tr>
      <tr><td class="non">Jour de la semaine</td><td class="non">Aucune</td></tr>
    </tbody>
  </table>
  </div>
  <p class="cle">
    L'enseignement principal est ailleurs : <b>l'essentiel de la variation reste
    inexpliqué</b> par ces éléments simples. Compter les caractères, les emojis ou
    les hashtags ne mène pas très loin. Ce qui fait la différence tient
    vraisemblablement à l'image elle-même et au sens du message, qui demandent des
    outils d'analyse plus avancés — c'est l'objet de la suite du projet.
  </p>
</section>

<footer>
  <p>
    <b>Méthode.</b> Données collectées via l'API officielle Instagram Graph, sans
    extraction sauvage. Engagement d'un post = (mentions J'aime + commentaires)
    rapportées au nombre d'abonnés du compte. Les publications dont le compteur de
    mentions J'aime est masqué par le compte ont été écartées.
  </p>
  <p>
    <b>Limites.</b> Le nombre d'abonnés n'est connu qu'à la date de collecte, ce qui
    pénalise les publications anciennes des comptes ayant beaucoup grandi. Les heures
    sont exprimées en temps universel et non dans le fuseau de chaque audience. Enfin,
    {comptes} comptes forment un échantillon restreint, centré sur les univers
    créatifs : ces résultats ne se transposent pas tels quels à d'autres secteurs.
  </p>
</footer>

</div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
