"""Palette et style partagés par le notebook d'exploration et le rapport HTML.

Palette de référence validée (checks lightness / chroma / CVD / normal-vision /
contraste). En mode clair, aqua, jaune et magenta passent sous 3:1 face au fond :
les graphiques qui les emploient portent des étiquettes de valeurs visibles.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Slots catégoriels, dans l'ordre validé. Ne jamais réordonner ni recycler.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
}
INK = {
    "light": {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
              "muted": "#8a8880", "grid": "#e4e3df"},
    "dark": {"surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
             "muted": "#8a8880", "grid": "#33332f"},
}

# Paire divergente (polarité au-dessus / en-dessous d'une référence), midpoint neutre.
DIVERGENT = {
    "light": {"pos": "#2a78d6", "neg": "#e34948", "neutre": "#f0efec"},
    "dark": {"pos": "#3987e5", "neg": "#e66767", "neutre": "#383835"},
}

THEMES = ["art", "ecriture", "melange", "nature", "photographie"]
TYPES_POST = ["image", "carrousel", "video"]


def couleurs_theme(mode="light"):
    return dict(zip(THEMES, SERIES[mode]))


def couleurs_type(mode="light"):
    return dict(zip(TYPES_POST, SERIES[mode][:3]))


def appliquer_style(mode="light"):
    ink = INK[mode]
    mpl.rcParams.update({
        "figure.facecolor": ink["surface"],
        "axes.facecolor": ink["surface"],
        "savefig.facecolor": ink["surface"],
        "text.color": ink["primary"],
        "axes.labelcolor": ink["secondary"],
        "axes.edgecolor": ink["grid"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": ink["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": ink["secondary"],
        "ytick.color": ink["secondary"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.dpi": 110,
    })
    return ink


def etiqueter_barres(ax, barres, valeurs, fmt="{:.1f}", mode="light", decalage=0.01):
    """Étiquettes de valeurs au-dessus des barres (règle de relief)."""
    ink = INK[mode]
    haut = max(valeurs) if len(valeurs) else 0
    for barre, v in zip(barres, valeurs):
        ax.annotate(fmt.format(v), (barre.get_x() + barre.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=ink["primary"])
    if haut:
        ax.set_ylim(top=haut * (1 + decalage + 0.08))


def barres_arrondies(ax, x, hauteurs, couleurs, largeur=0.62, mode="light"):
    """Barres fines, extrémité arrondie, ancrées à la ligne de base."""
    ink = INK[mode]
    barres = ax.bar(x, hauteurs, width=largeur, color=couleurs,
                    edgecolor=ink["surface"], linewidth=2)
    return barres


def finir(ax, titre=None, sous_titre=None, ylabel=None, xlabel=None, mode="light"):
    ink = INK[mode]
    if titre:
        # Le sous-titre occupe la bande juste au-dessus des axes : on pousse le
        # titre plus haut pour qu'ils ne se chevauchent pas.
        ax.set_title(titre, color=ink["primary"], pad=30 if sous_titre else 12)
    if sous_titre:
        ax.annotate(sous_titre, xy=(0, 1.015), xycoords="axes fraction",
                    fontsize=9, color=ink["secondary"], va="bottom")
    ax.set_ylabel(ylabel or "", color=ink["secondary"])
    ax.set_xlabel(xlabel or "", color=ink["secondary"])
    ax.grid(axis="x", visible=False)
    return ax
