# Prédire l'engagement d'un post Instagram

Un post publié aujourd'hui fera-t-il mieux ou moins bien que ce que ce compte obtient
d'habitude ? Ce projet répond à cette question **avant publication**, à partir de
l'image, de la légende et de l'heure prévue, en combinant vision par ordinateur,
traitement du langage et analyse temporelle.

**Résultat principal : le modèle capte un signal réel mais faible — AUC 0,524 à 0,530
contre 0,500 au hasard.** Ce chiffre est le cœur du projet autant que son aboutissement :
la démarche a consisté à mesurer honnêtement ce qui était prédictible, et à identifier
précisément ce qui ne l'était pas.

| | |
|---|---|
| Données | 4 173 publications, 29 comptes, collectées via l'API Instagram Graph |
| Jeu modélisé | 3 307 publications, 28 comptes, 255 à 11 209 abonnés |
| Variables | embeddings CLIP (image), embeddings multilingues (texte), temporel |
| Livrable | application Streamlit affichant une tendance et sa fiabilité réelle |

**Parcours rapide :** [notebook d'exploration](notebooks/01_exploration.ipynb) ·
[notebook de modélisation](notebooks/02_modelisation.ipynb) ·
[notebook final](notebooks/03_modelisation_relative.ipynb) ·
[rapport visuel non technique](reports/exploration_summary.html) ·
[journal technique détaillé](docs/journal-des-phases.md)

---

## Pourquoi une cible relative au compte

La formulation évidente — « combien de mentions J'aime ce post va-t-il récolter ? » — a
été testée, puis abandonnée sur la base d'une mesure.

La décomposition de la variance montre que **44 % de ce qu'il faudrait prédire tient à
l'identité du compte**, pas au contenu du post. Avec seulement 21 comptes
d'entraînement, apprendre à généraliser d'un compte à l'autre n'était pas réaliste. Le
diagnostic s'est confirmé sans ambiguïté : aucun modèle n'a battu une régression
linéaire sur le seul nombre d'abonnés.

La cible est donc devenue relative :

```
cible = log(1 + interactions) − log(1 + niveau habituel du compte)
```

où le **niveau habituel** est la médiane des interactions des 20 publications
précédentes du même compte. Le décalage d'un rang rend la cible utilisable en
prédiction : au moment de publier, seul le passé est connu.

Ce changement n'est pas un repli. C'est aussi la question que se pose réellement un
créateur, qui connaît son niveau habituel et veut savoir si *ce* post fera mieux.

---

## Deux pivots qui ont changé le projet

### Pivot 1 — Un modèle de texte anglophone sur un corpus à 29 % arabe

Le plan initial prévoyait `all-MiniLM-L6-v2`, le modèle d'embeddings de référence pour
sa légèreté. Un contrôle avant entraînement a révélé que **28,7 % des légendes sont en
écriture non latine**, dont 996 en arabe, et que le thème *écriture* l'est à 97 %.

Test sur quatre phrases arabes — deux sur l'amour, deux sur la mer — en mesurant l'écart
entre similarité intra-sujet et inter-sujet :

| Modèle | Pouvoir de séparation |
|---|---|
| `all-MiniLM-L6-v2` (anglophone) | **−0,04** |
| `paraphrase-multilingual-MiniLM-L12-v2` | **+0,60** |

Un pouvoir de séparation négatif signifie que le modèle juge « amour » et « mer » *plus*
proches que deux phrases synonymes. Ses vecteurs auraient été du bruit pur pour près
d'un tiers du jeu de données, éliminant silencieusement un thème entier — sans qu'aucune
métrique d'entraînement ne le signale.

Le modèle multilingue sort la même dimension (384) pour un coût comparable. Vérification
après coup sur le jeu réel : les légendes arabes se regroupent à 0,528 de similarité
interne, contre 0,341 face aux légendes latines d'un autre thème.

**Ce que ce pivot illustre :** un contrôle de cinq minutes sur la validité d'un outil
pré-entraîné, avant de lancer quinze minutes de calcul et toute une chaîne de
modélisation dessus.

### Pivot 2 — De la valeur absolue à l'écart au compte

La phase de modélisation a produit un résultat sans appel :

| Modèle | RMSE (log) |
|---|---|
| Baseline : médiane du train | 1,402 |
| **Baseline : régression sur le nombre d'abonnés seul** | **1,362** |
| LightGBM réglé, 911 variables | 1,368 |
| LightGBM, meilleure variante (texte seul) | 1,365 |

Trois algorithmes, 911 variables, un réglage par recherche aléatoire validée par
`GroupKFold` : le meilleur score reste au niveau d'une droite ajustée sur un seul nombre.
L'erreur typique correspondait à un **facteur 3** — un post à 100 réactions prédit entre
33 et 300.

Plutôt que d'accepter le résultat, l'analyse a cherché la cause. Trois mesures l'ont
établie :

- **L'ablation** : les 15 variables tabulaires seules font pire que la médiane (1,533).
- **L'importance par permutation**, mesurée sur la validation : la plupart des blocs ont
  une contribution **négative**, c'est-à-dire que les mélanger *améliore* la prédiction.
  Le modèle s'appuyait dessus d'une façon qui ne se transposait pas à des comptes jamais
  vus.
- **La décomposition de la variance** : 44 % entre comptes, pour 21 comptes
  d'entraînement.

Le problème n'était ni l'algorithme ni les variables, mais **la question posée**. D'où
la cible relative — et un signal est apparu là où il n'y en avait aucun.

---

## Résultats

### Le modèle

Validation croisée groupée par compte, 5 folds sur les 28 comptes. Chaque fold met de
côté des comptes entiers, jamais des publications isolées.

| Contexte | AUC | Où il s'applique |
|---|---|---|
| Centrage par compte, posts précédents | **0,530** | borne haute, utilisateur disposant d'un historique |
| Centrage sur la moyenne du jeu | **0,524** | **chiffre affiché par l'application** |
| Hasard | 0,500 | référence |

Les deux chiffres correspondent à deux situations réelles. À l'entraînement, chaque
image et chaque légende étaient comparées au style habituel de leur compte. Un nouvel
utilisateur n'ayant pas cet historique, l'application compare au style moyen du jeu
d'entraînement — et le coût de cette approximation a été mesuré, pas supposé.

Les quatre familles de modèles testées (Ridge, RandomForest, XGBoost, LightGBM)
dépassent le hasard dans les cinq folds, avec les deux méthodes de centrage. La
régularité est ce qui rend le signal crédible.

**Ce que ce chiffre veut dire concrètement :** face à deux publications dont l'une a fait
mieux que l'habitude et l'autre moins bien, le modèle désigne la bonne dans 52 à 53 % des
cas, contre 50 % à pile ou face. C'est mesurable et reproductible. Ce n'est pas
suffisant pour décider quoi publier, et l'application le dit.

### Les facteurs descriptifs, eux, sont solides

L'analyse exploratoire de 3 587 publications a produit des constats qui ne dépendent
d'aucun modèle prédictif. Chaque publication y est comparée à la médiane de son propre
compte, pour que la notoriété ne fausse pas la lecture.

**L'heure de publication compte, modérément.** Le pic se situe à **9 h UTC, avec +25 %**
par rapport à l'ordinaire du compte. L'écart entre le meilleur et le pire créneau est
d'un facteur 1,7 (p < 0,001).

**Les hashtags sont utiles en petit nombre, inutiles en pagaille.**

| Hashtags | Écart au niveau habituel |
|---|---|
| aucun | −8 % |
| **1 à 5** | **+9 %** |
| 6 à 10 | −5 % |
| 11 à 20 | −3 % |
| 21 et plus | −4 % |

L'effet est en cloche, ce qui explique qu'une corrélation linéaire le manque entièrement
(rho = −0,009). Seul un découpage par tranches le révèle, et il survit au contrôle par
compte.

**Et deux conseils répandus qui ne tiennent pas :** le jour de la semaine n'a aucun effet
(p = 0,86), la présence d'emojis non plus en pratique (+1 %, négligeable malgré
p = 0,003).

### Une nuance méthodologique qui vaut d'être notée

L'heure de publication a un effet descriptif net (facteur 1,7) mais une importance quasi
nulle dans le modèle (0,0003). Ce n'est pas une contradiction : l'heure explique **0,58 %
de la variance totale** à prédire. L'effet est mesuré à l'intérieur de chaque compte, où
+25 % est visible et actionnable ; le modèle travaille sur une échelle où il est noyé.

**Un effet peut être statistiquement solide, pratiquement utile pour un créateur, et
malgré tout négligeable pour un modèle de prédiction.** L'ordre de grandeur compte autant
que la significativité.

---

## Limites

Ces limites ne sont pas des réserves de politesse : ce sont les raisons mesurées pour
lesquelles le signal plafonne à 0,53.

**Les vidéos ne sont vues que par leur miniature.** 56 % du jeu est constitué de reels,
dont seule l'image de couverture est analysée. Le mouvement, le rythme, le montage et le
son — c'est-à-dire l'essentiel de ce qui fait marcher un reel — sont totalement absents
du modèle. L'ablation le confirme : les embeddings d'image n'apportent rien et dégradent
même légèrement le score lorsqu'on les ajoute au texte.

**Les carrousels ne sont vus que par leur première image**, alors qu'ils peuvent en
compter jusqu'à dix.

**L'échantillon est étroit.** 28 comptes de 255 à 11 209 abonnés, tous dans des univers
créatifs : art, photographie, écriture arabe, nature. Rien ne garantit la transposition à
d'autres secteurs ou à des comptes beaucoup plus suivis. Le compte de référence du
projet, avec ses **48 abonnés, se situe sous le plancher** de cette fourchette : les
estimations le concernant sont moins fiables encore.

**L'essentiel de la variation échappe au contenu.** L'écart-type de la cible vaut 1,12,
soit un **facteur 3** entre deux publications d'un même compte. Une large part tient à la
diffusion algorithmique d'Instagram, que rien dans une image ou une légende ne permet
d'anticiper. Aucun modèle testé ne bat le pari trivial « ce post fera comme d'habitude »
sur l'erreur de prédiction — c'est précisément pourquoi l'application n'affiche aucun
chiffre.

**Tout ce qui précède décrit des associations, pas des causalités.** Rien n'établit que
publier à 9 h *provoque* plus de réactions ; les comptes qui publient au bon moment
soignent vraisemblablement aussi le reste.

---

## Pistes d'amélioration

Par impact attendu décroissant.

**1. Collecter davantage de comptes.** C'est le vrai levier, identifié par la
décomposition de la variance : le problème est le nombre de *comptes*, pas de
publications. Doubler le nombre de posts par compte n'aiderait pas ; passer de 28 à 60
comptes, si. La collecte étant automatisée, c'est surtout une question de temps et de
quotas d'API.

**2. Centrer les embeddings sur l'historique réel de l'utilisateur.** L'application
compare aujourd'hui au style moyen du jeu d'entraînement, faute d'historique. Permettre
à un utilisateur de fournir ses publications passées récupérerait l'écart mesuré entre
0,524 et 0,530 — et probablement davantage, un historique réel valant mieux qu'une
moyenne de population.

**3. Extraire plusieurs images par vidéo.** Trois à cinq images réparties dans un reel
plutôt qu'une miniature figée donneraient au modèle une chance de capter le rythme et la
variété visuelle, sur 56 % du jeu actuellement mal représenté.

**4. Enrichir le jeu de petits comptes.** La comparaison menée en phase 4 a montré qu'un
modèle entraîné sur les seuls comptes de moins de 2 000 abonnés généralise *moins* bien
(RMSE 1,472 contre 1,348) : la perte de volume l'emporte sur la spécialisation. Cette
conclusion repose toutefois sur 3 comptes de validation seulement et mérite d'être
retestée sur un échantillon élargi.

---

## Structure du dépôt

```
.
├── app.py                          application Streamlit
├── src/
│   ├── collection/                 phase 1 — API Instagram Graph
│   │   ├── client.py               Business Discovery, pagination, limites de débit
│   │   ├── collect.py              collecte reprenable
│   │   └── download_media.py       téléchargement des images
│   ├── processing/clean_data.py    phase 2 — dataset propre, découpage par compte
│   ├── features/build_features.py  phase 3 — embeddings CLIP + texte, temporel
│   ├── models/
│   │   ├── train_models.py         phase 4 — baselines, 3 candidats, ablation
│   │   ├── train_relatif.py        phase 4b — cible relative, ACP, CV groupée
│   │   └── export_app.py           artefacts pour l'application
│   └── viz/                        palette validée, rapport HTML
├── notebooks/                      exploration et modélisation, exécutés
├── reports/                        rapport visuel non technique
├── models/app/                     artefacts versionnés (302 Ko)
└── docs/journal-des-phases.md      journal technique détaillé
```

## Installation et exécution

```bash
pip install -r requirements.txt
```

Créer un fichier `.env` à la racine (non versionné) :

| Variable | Rôle |
|---|---|
| `IG_ACCESS_TOKEN` | Token longue durée d'un compte Business/Creator lié à une Page Facebook. Permissions : `instagram_basic`, `instagram_manage_insights`, `pages_read_engagement`, `pages_show_list` |
| `IG_USER_ID` | Identifiant du compte émettant les appels Business Discovery |
| `GRAPH_API_VERSION` | Version de l'API, `v21.0` par défaut |

```bash
# Phase 1 — collecte (nécessite l'API)
python -m src.collection.collect              # comptes de data/raw/accounts.csv
python -m src.collection.collect --own        # compte personnel
python -m src.collection.download_media       # images : les URLs expirent vite

# Phase 2 — nettoyage et rapport
python -m src.processing.clean_data
python -m src.viz.build_report

# Phase 3 — variables (~15 min, puis 4 s grâce au cache)
python -m src.features.build_features

# Phases 4 et 4b — modélisation
python -m src.models.train_models
python -m src.models.train_relatif

# Phase 5 — application
python -m src.models.export_app
streamlit run app.py
```

L'application démarre depuis un clone sans refaire la chaîne : ses artefacts sont
versionnés et les agrégats descriptifs sont embarqués. Les modèles pré-entraînés se
téléchargent au premier lancement.

## Stack

| Domaine | Outils |
|---|---|
| Collecte | API Instagram Graph (Business Discovery), `requests` |
| Vision | CLIP `openai/clip-vit-base-patch32` via `transformers` |
| Texte | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Modélisation | scikit-learn (Ridge, RandomForest, ACP), XGBoost, LightGBM |
| Analyse | pandas, NumPy, SciPy, matplotlib |
| Application | Streamlit |

## Notes sur les données

Les données brutes et le jeu nettoyé ne sont pas versionnés : ils contiennent le contenu
de comptes tiers. Tout se régénère depuis l'API avec les commandes ci-dessus. Seuls les
agrégats et les artefacts du modèle, qui ne permettent pas de reconstituer les
publications d'origine, figurent dans le dépôt.

La collecte passe exclusivement par l'API officielle, sans extraction sauvage.
