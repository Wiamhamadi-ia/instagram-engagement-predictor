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

from src.viz.style import (DIVERGENT, INK, TYPES_POST, appliquer_style,
                           couleurs_theme, couleurs_type)

SOURCE = Path("data/processed/dataset_clean.csv")
SORTIE = Path("reports/exploration_summary.html")
JOURS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
JOURS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SEUIL_HEURE_FIABLE = 60


def en_svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue().encode("utf-8")).decode("ascii")


def titre_axe(ax, texte, ink):
    ax.set_title(texte, color=ink["primary"], fontsize=11, pad=10)


def barres_ecart(ax, valeurs, etiquettes, mode, fmt="{:+.0f} %"):
    """Barres ancrées au niveau habituel du compte, colorées par polarité."""
    ink, div = INK[mode], DIVERGENT[mode]
    b = ax.bar(range(len(valeurs)), valeurs, width=0.6,
               color=[div["pos"] if v >= 0 else div["neg"] for v in valeurs],
               edgecolor=ink["surface"], linewidth=2)
    ax.axhline(0, color=ink["primary"], linewidth=1.2)
    for barre, v in zip(b, valeurs):
        ax.annotate(fmt.format(v * 100), (barre.get_x() + barre.get_width() / 2, v),
                    xytext=(0, 4 if v >= 0 else -15), textcoords="offset points",
                    ha="center", fontsize=10, color=ink["primary"], fontweight="semibold")
    ax.set_xticks(range(len(etiquettes)))
    ax.set_xticklabels(etiquettes)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.32)
    return b


# --- graphiques ---------------------------------------------------------------

def fig_distribution(u, mode):
    ink, c = INK[mode], couleurs_theme(mode)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    pos = u.interactions[u.interactions > 0]
    ax.hist(pos, bins=np.logspace(0, np.log10(pos.max()), 45), color=c["art"], edgecolor="none")
    ax.set_xscale("log")
    med = u.interactions.median()
    ax.axvline(med, color=ink["primary"], linestyle="--", linewidth=1.5)
    ax.annotate(f"moitié des posts\nsous {med:.0f} réactions", (med, ax.get_ylim()[1] * 0.95),
                xytext=(10, 0), textcoords="offset points", va="top",
                fontsize=9, color=ink["primary"], fontweight="semibold")
    ax.set_xlabel("réactions reçues (mentions J'aime + commentaires)", color=ink["secondary"])
    ax.set_ylabel("nombre de posts", color=ink["secondary"])
    ax.grid(axis="x", visible=False)
    return en_svg(fig)


def fig_heure(u, mode):
    ink, c = INK[mode], couleurs_theme(mode)
    h = u.groupby("heure").agg(n=("id", "count"), rel=("rel", "median"))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(h.index, h.rel, color=c["art"], marker="o", markersize=8,
            markeredgecolor=ink["surface"], markeredgewidth=2, solid_capstyle="round")
    ax.axhline(1.0, color=ink["primary"], linestyle="--", linewidth=1.2)

    maigre = h[h.n < SEUIL_HEURE_FIABLE]
    ax.scatter(maigre.index, maigre.rel, s=170, facecolor="none",
               edgecolor=ink["muted"], linewidth=2, zorder=4)
    ax.annotate("cercles gris = créneaux trop peu fournis pour conclure",
                xy=(0.99, 0.03), xycoords="axes fraction", ha="right",
                fontsize=8, color=ink["secondary"])

    pic = h.rel.idxmax()
    ax.annotate(f"+{(h.rel.max() - 1) * 100:.0f} % à {pic} h", (pic, h.rel.max()),
                xytext=(0, 11), textcoords="offset points", ha="center",
                fontsize=10, color=ink["primary"], fontweight="semibold")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{k} h" for k in range(0, 24, 3)])
    ax.set_yticks([0.8, 1.0, 1.2])
    ax.set_yticklabels(["−20 %", "niveau habituel", "+20 %"])
    ax.margins(y=0.2)
    ax.grid(axis="x", visible=False)
    return en_svg(fig)


def fig_format(u, mode):
    """Vue brute contre vue corrigée : l'ordre s'inverse."""
    ink, c = INK[mode], couleurs_type(mode)
    stat = u.groupby("type_post").agg(med=("interactions", "median"),
                                      rel=("rel", "median")).reindex(TYPES_POST)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))

    b = axes[0].bar(range(3), stat.med, width=0.6, color=[c[t] for t in TYPES_POST],
                    edgecolor=ink["surface"], linewidth=2)
    for barre, v in zip(b, stat.med):
        axes[0].annotate(f"{v:.0f}", (barre.get_x() + barre.get_width() / 2, v),
                         xytext=(0, 4), textcoords="offset points", ha="center",
                         fontsize=10, color=ink["primary"], fontweight="semibold")
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(TYPES_POST)
    axes[0].set_ylim(top=stat.med.max() * 1.22)
    axes[0].set_ylabel("réactions", color=ink["secondary"])
    axes[0].grid(axis="x", visible=False)
    titre_axe(axes[0], "Tous comptes mélangés", ink)

    barres_ecart(axes[1], (stat.rel - 1).values, TYPES_POST, mode)
    axes[1].set_yticks([-0.05, 0, 0.05])
    axes[1].set_yticklabels(["−5 %", "niveau habituel", "+5 %"])
    titre_axe(axes[1], "Chaque compte comparé à lui-même", ink)
    return en_svg(fig)


def fig_hashtags(u, mode):
    ink = INK[mode]
    noms = ["aucun", "1 à 5", "6 à 10", "11 à 20", "21 et +"]
    tr = pd.cut(u.nb_hashtags, [-1, 0, 5, 10, 20, 1000], labels=noms)
    g = u.groupby(tr, observed=True).rel.median().reindex(noms) - 1.0
    n = u.groupby(tr, observed=True).size().reindex(noms)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    barres_ecart(ax, g.values, [f"{t}\n{int(k)} posts" for t, k in zip(noms, n)], mode)
    ax.set_yticks([-0.05, 0, 0.05, 0.10])
    ax.set_yticklabels(["−5 %", "niveau habituel", "+5 %", "+10 %"])
    ax.set_xlabel("nombre de hashtags", color=ink["secondary"])
    return en_svg(fig)


def fig_jour(u, mode):
    j = u.groupby("nom_jour").rel.median().reindex(JOURS_EN) - 1.0
    fig, ax = plt.subplots(figsize=(8, 3.2))
    barres_ecart(ax, j.values, JOURS_FR, mode, fmt="{:+.1f} %")
    ax.set_yticks([-0.05, 0, 0.05])
    ax.set_yticklabels(["−5 %", "niveau habituel", "+5 %"])
    return en_svg(fig)


def fig_taille_compte(u, mode):
    ink, c = INK[mode], couleurs_theme(mode)
    parc = u.groupby("compte").agg(followers=("followers", "first"),
                                   med=("interactions", "median"))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.scatter(parc.followers, parc.med, s=95, color=c["art"],
               edgecolor=ink["surface"], linewidth=2, zorder=3)
    ordre = parc.sort_values("followers")
    lisse = ordre.med.rolling(7, center=True, min_periods=3).median()
    ax.plot(ordre.followers, lisse, color=ink["muted"], linewidth=2, linestyle="--", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(parc.followers.min() * 0.75, parc.followers.max() * 1.5)
    ax.set_xlabel("abonnés du compte", color=ink["secondary"])
    ax.set_ylabel("réactions par post", color=ink["secondary"])
    ax.annotate("tendance", (ordre.followers.iloc[-4], lisse.iloc[-4]),
                xytext=(6, 10), textcoords="offset points",
                fontsize=9, color=ink["secondary"])
    ax.grid(axis="x", visible=False)
    return en_svg(fig)


def fig_themes(u, mode):
    ink, c = INK[mode], couleurs_theme(mode)
    med = u.groupby("theme").interactions.median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(range(len(med)), med.values, height=0.66,
            color=[c[t] for t in med.index], edgecolor=ink["surface"], linewidth=2)
    for i, v in enumerate(med.values):
        ax.annotate(f"{v:.0f}", (v, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, color=ink["primary"], fontweight="semibold")
    ax.set_yticks(range(len(med))); ax.set_yticklabels(med.index)
    ax.invert_yaxis()
    ax.set_xlim(0, med.max() * 1.2)
    ax.set_xlabel("réactions par post", color=ink["secondary"])
    ax.grid(axis="y", visible=False)
    return en_svg(fig)


FIGURES = [
    ("distribution", fig_distribution), ("heure", fig_heure), ("format", fig_format),
    ("hashtags", fig_hashtags), ("jour", fig_jour),
    ("taille", fig_taille_compte), ("themes", fig_themes),
]


def bloc_figure(svgs, nom, alt):
    return (f'<figure class="chart">'
            f'<img class="only-light" alt="{alt}" src="data:image/svg+xml;base64,{svgs[nom + "_light"]}">'
            f'<img class="only-dark" alt="{alt}" src="data:image/svg+xml;base64,{svgs[nom + "_dark"]}">'
            f"</figure>")


def main():
    df = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    u = df[df.utilisable_entrainement].copy()
    u["rel"] = u.interactions / u.groupby("compte").interactions.transform("median")

    svgs = {}
    for mode in ("light", "dark"):
        appliquer_style(mode)
        for nom, fn in FIGURES:
            svgs[f"{nom}_{mode}"] = fn(u, mode)

    h = u.groupby("heure").agg(n=("id", "count"), rel=("rel", "median"))
    h_fiable = h[h.n >= SEUIL_HEURE_FIABLE]
    fmt = u.groupby("type_post").agg(med=("interactions", "median"),
                                     moy=("interactions", "mean"), rel=("rel", "median"))
    med_theme = u.groupby("theme").interactions.median().sort_values(ascending=False)
    ht = u.groupby(pd.cut(u.nb_hashtags, [-1, 0, 5, 10, 20, 1000]), observed=True).rel.median()

    kpi = {
        "posts": f"{len(u):,}".replace(",", " "),
        "comptes": u.compte.nunique(),
        "median": f"{u.interactions.median():.0f}",
        "moyenne": f"{u.interactions.mean():.0f}",
        "max": f"{u.interactions.max():,.0f}".replace(",", " "),
        "gros": f"{(u.interactions > 1000).mean() * 100:.0f} %",
        "pic": f"{h_fiable.rel.idxmax()} h",
        "pic_gain": f"{(h_fiable.rel.max() - 1) * 100:.0f}",
        "ratio_heure": f"{h.rel.max() / h.rel.min():.1f}",
        "video_med": f"{fmt.loc['video', 'med']:.0f}",
        "video_moy": f"{fmt.loc['video', 'moy']:.0f}",
        "image_med": f"{fmt.loc['image', 'med']:.0f}",
        "carrousel_rel": f"{(fmt.loc['carrousel', 'rel'] - 1) * 100:+.0f}",
        "image_rel": f"{(fmt.loc['image', 'rel'] - 1) * 100:+.0f}",
        "ht_gain": f"{(ht.iloc[1] - 1) * 100:+.0f}",
        "theme_haut": med_theme.index[0], "theme_haut_v": f"{med_theme.iloc[0]:.0f}",
        "theme_bas": med_theme.index[-1], "theme_bas_v": f"{med_theme.iloc[-1]:.0f}",
        "debut": u.timestamp.min().year, "fin": u.timestamp.max().year,
        "date": date.today().strftime("%d/%m/%Y"),
    }

    html = GABARIT.format(**kpi, **{f"fig_{n}": bloc_figure(svgs, n, a) for n, a in [
        ("distribution", "Répartition des réactions reçues par les publications"),
        ("heure", "Réactions selon l'heure de publication"),
        ("format", "Réactions par format de publication"),
        ("hashtags", "Effet du nombre de hashtags"),
        ("jour", "Réactions selon le jour de publication"),
        ("taille", "Réactions selon la taille du compte"),
        ("themes", "Réactions par thème de compte"),
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
  .methode {{ border: 1px solid var(--border); border-radius: 12px;
              padding: 18px 20px; margin-bottom: 48px; background: var(--card); }}
  .methode h3 {{ margin: 0 0 8px; font-size: 16px; }}
  .methode p {{ margin: 0; color: var(--ink-2); font-size: 15px; }}
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
    {posts} publications de {comptes} comptes créatifs, parues entre {debut} et {fin},
    passées au crible. Objectif : distinguer ce qui influence vraiment les réactions
    d'un post des conseils que tout le monde répète.
  </p>
  <p class="meta">Analyse réalisée le {date} · source : API Instagram Graph</p>
</header>

<ul class="kpis">
  <li><b>{posts}</b><span>publications analysées</span></li>
  <li><b>{comptes}</b><span>comptes suivis</span></li>
  <li><b>{median}</b><span>réactions pour un post typique</span></li>
  <li><b>{gros}</b><span>de posts dépassant 1 000 réactions</span></li>
</ul>

<div class="methode">
  <h3>Comment lire les chiffres qui suivent</h3>
  <p>
    On compte les <b>réactions</b> reçues par une publication : mentions J'aime plus
    commentaires. Un gros compte en récolte naturellement plus qu'un petit. Pour éviter
    de mesurer la notoriété plutôt que le contenu, la plupart des graphiques comparent
    <b>chaque publication à ce que son propre compte obtient d'habitude</b>. « +10 % »
    signifie donc « 10 % de mieux que l'ordinaire de ce compte-là », et non « 10 % de
    mieux que les autres comptes ».
  </p>
</div>

<section>
  <h2>Un post ordinaire fait peu de bruit, quelques-uns explosent</h2>
  <p>
    Chaque barre compte les publications ayant obtenu un nombre de réactions donné.
    L'échelle horizontale est resserrée : chaque graduation vaut dix fois la précédente,
    sinon la quasi-totalité des posts s'entasserait à gauche.
  </p>
  {fig_distribution}
  <p class="cle">
    La moitié des publications reste sous <b>{median} réactions</b>, mais la moyenne
    grimpe à {moyenne}, soit treize fois plus, et le record atteint {max}. Une poignée de
    posts viraux écrase tout le reste. <b>Parler de « moyenne » n'a donc aucun sens
    ici</b> : tous les chiffres de cette page décrivent le post typique, pas la moyenne.
  </p>
</section>

<section>
  <h2>L'heure de publication compte, mais moins qu'on ne le dit</h2>
  <p>
    Écart au niveau habituel du compte, selon l'heure de mise en ligne, en temps
    universel (UTC).
  </p>
  {fig_heure}
  <p class="cle">
    Le créneau de <b>{pic} apporte {pic_gain} %</b> de réactions de plus que l'ordinaire
    du compte. L'effet est bien réel, mais son ampleur reste mesurée : un facteur
    {ratio_heure} entre le pire et le meilleur créneau, pas davantage. Les heures de nuit,
    entourées en gris, reposent sur trop peu de publications pour qu'on puisse en tirer
    quoi que ce soit.
  </p>
</section>

<section>
  <h2>Le format : deux lectures opposées</h2>
  <p>
    À gauche, tous les comptes sont mélangés. À droite, chaque publication est comparée
    à son propre compte.
  </p>
  {fig_format}
  <p class="cle">
    Les deux vues se contredisent, et c'est tout l'intérêt. Mélangés, les posts photo
    semblent gagner ({image_med} réactions contre {video_med} pour la vidéo). Une fois
    chaque compte comparé à lui-même, l'ordre s'inverse : le <b>carrousel passe devant
    ({carrousel_rel} %)</b> et la photo seule tombe dernière ({image_rel} %). La première
    lecture reflétait surtout le fait que les photos viennent de comptes plus actifs.
  </p>
</section>

<section>
  <h2>La vidéo est un billet de loterie</h2>
  <p>
    Un constat qui ne tient pas dans un graphique, mais qui saute aux yeux dans les
    chiffres bruts.
  </p>
  <p class="cle">
    Une vidéo ordinaire récolte <b>{video_med} réactions</b>, moins qu'une photo
    ({image_med}). Mais la vidéo moyenne en récolte <b>{video_moy}</b>, soit vingt fois
    sa propre médiane : les rares vidéos virales concentrent l'essentiel des réactions.
    La vidéo augmente les chances de carton tout en abaissant le résultat le plus
    probable.
  </p>
</section>

<section>
  <h2>Les hashtags : utiles en petit nombre, inutiles en pagaille</h2>
  <p>Écart au niveau habituel du compte, selon le nombre de hashtags dans la légende.</p>
  {fig_hashtags}
  <p class="cle">
    Les publications portant <b>un à cinq hashtags dépassent de {ht_gain} %</b>
    l'ordinaire de leur compte. Au-delà, le bénéfice disparaît : à partir de six, et
    jusqu'à vingt et plus, les posts repassent sous ce niveau. <b>En empiler trente
    n'apporte rien.</b>
  </p>
</section>

<section>
  <h2>Le jour de la semaine, lui, ne change rien</h2>
  <p>Écart au niveau habituel du compte, selon le jour de mise en ligne.</p>
  {fig_jour}
  <p class="cle">
    Les sept barres tiennent dans un mouchoir de poche, à quelques pourcents du niveau
    de référence. Un mardi vaut un samedi. <b>L'heure compte un peu, le jour pas du
    tout.</b>
  </p>
</section>

<section>
  <h2>Les petits comptes ne sont pas hors course</h2>
  <p>Chaque point représente un compte : ses abonnés et ses réactions habituelles.</p>
  {fig_taille}
  <p class="cle">
    Avoir plus d'abonnés aide, sans surprise. Mais le lien est <b>bien plus lâche qu'on
    ne l'imaginerait</b> : à nombre d'abonnés comparable, les écarts entre comptes sont
    énormes, et certains petits comptes dépassent des comptes dix fois plus suivis.
    La part de l'audience qui réagit diminue quand le compte grandit, ce qui compense
    en partie l'effet de taille. <b>Le contenu garde donc une vraie marge de manœuvre.</b>
  </p>
</section>

<section>
  <h2>Le sujet du compte pèse, mais reste mêlé à sa taille</h2>
  <p>Réactions du post typique selon la thématique dominante du compte.</p>
  {fig_themes}
  <p class="cle">
    L'écart va de <b>{theme_bas_v}</b> réactions ({theme_bas}) à <b>{theme_haut_v}</b>
    ({theme_haut}). Ce classement est à prendre avec précaution : les thématiques ne
    rassemblent pas des comptes de taille comparable, si bien que l'on mesure ici un
    mélange de sujet et de notoriété.
  </p>
</section>

<section>
  <h2>Ce qui compte, ce qui ne compte pas</h2>
  <div class="tablewrap">
  <table>
    <thead><tr><th>Facteur</th><th>Influence mesurée</th></tr></thead>
    <tbody>
      <tr><td class="oui">Taille du compte</td><td class="oui">Réelle, mais plus faible qu'attendu</td></tr>
      <tr><td class="oui">Heure de publication</td><td class="oui">Modérée — facteur {ratio_heure}</td></tr>
      <tr><td class="oui">Format du post</td><td class="oui">Modérée — carrousel devant</td></tr>
      <tr><td class="oui">Thème du compte</td><td class="oui">Modérée — mêlée à la taille</td></tr>
      <tr><td class="oui">Hashtags (1 à 5)</td><td class="oui">Faible — environ {ht_gain} %</td></tr>
      <tr><td class="non">Longueur de la légende</td><td class="non">Négligeable</td></tr>
      <tr><td class="non">Présence d'emojis</td><td class="non">Négligeable</td></tr>
      <tr><td class="non">Jour de la semaine</td><td class="non">Aucune</td></tr>
    </tbody>
  </table>
  </div>
  <p class="cle">
    L'enseignement principal est ailleurs : <b>l'essentiel de la variation reste
    inexpliqué</b> par ces éléments simples. Compter les caractères, les emojis ou les
    hashtags ne mène pas très loin. Ce qui fait la différence tient vraisemblablement à
    l'image elle-même et au sens du message, qui demandent des outils d'analyse plus
    avancés — c'est l'objet de la suite du projet.
  </p>
</section>

<footer>
  <p>
    <b>Méthode.</b> Données collectées via l'API officielle Instagram Graph, sans
    extraction sauvage. On mesure les réactions d'un post, mentions J'aime plus
    commentaires, et non un taux rapporté aux abonnés : le projet vise les petits
    comptes, pour lesquels diviser par une audience minuscule produit surtout du bruit.
    Les publications dont le compteur de mentions J'aime est masqué par le compte ont été
    écartées, faute de valeur connue. Celles qui ont réellement fait zéro ont été
    conservées : un échec est une information.
  </p>
  <p>
    <b>Limites.</b> Les heures sont exprimées en temps universel et non dans le fuseau de
    chaque audience. Rien ici ne démontre de lien de cause à effet : publier à 9 h ou
    ajouter trois hashtags n'est pas prouvé <i>provoquer</i> plus de réactions, les
    comptes qui font l'un soignant vraisemblablement aussi le reste. Enfin, {comptes}
    comptes de 255 à 11 209 abonnés forment un échantillon restreint, centré sur les
    univers créatifs : ces résultats ne se transposent pas tels quels à d'autres secteurs
    ni à des comptes beaucoup plus suivis.
  </p>
</footer>

</div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
