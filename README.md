# Prédiction d'engagement Instagram

Prédire l'engagement d'un post Instagram avant publication (image + légende + heure), avec CV, NLP et séries temporelles, puis une app Streamlit.

## Phases
1. **Collecte** : API Instagram Graph (Business Discovery), 28 comptes publics + mon compte
2. **Exploration & nettoyage** : dataset propre, notebook d'analyse, rapport visuel
3. Feature engineering (CV, NLP, temporel) — *en cours*
4. Modélisation (baseline puis XGBoost/LightGBM)
5. App Streamlit
6. Storytelling (lecture Analyst / Data Scientist)

## Phase 1 : collecte
Créer un fichier `.env` (non versionné) à la racine :

| Variable | Rôle |
|---|---|
| `IG_ACCESS_TOKEN` | Token longue durée d'un compte Business/Creator lié à une Page Facebook. Permissions requises : `instagram_basic`, `instagram_manage_insights`, `pages_read_engagement`, `pages_show_list`. |
| `IG_USER_ID` | ID du compte Instagram Business qui émet les appels Business Discovery |
| `GRAPH_API_VERSION` | Version de l'API, `v21.0` par défaut |

```
pip install -r requirements.txt
python -m src.collection.collect            # comptes de data/raw/accounts.csv
python -m src.collection.collect --own      # mon compte
```
```
python -m src.collection.download_media             # images (URLs expirent vite)
```
Sorties dans `data/raw/` : `posts.jsonl`, `profiles.jsonl`, `failed_accounts.csv`,
`media_index.csv`, `media/<compte>/<id>.jpg`.

### Résultat
4 173 posts sur 29 comptes (28 comptes similaires + mon compte), publiés de 2013 à 2026.
4 169 images téléchargées (miniature pour les vidéos, image principale sinon).

| Thème | Comptes | Posts | Vidéos | Carrousels | Images | Sans likes |
|---|---|---|---|---|---|---|
| nature | 6 | 1 104 | 715 | 198 | 191 | 205 |
| art | 6 | 916 | 291 | 492 | 133 | 60 |
| ecriture | 6 | 860 | 532 | 40 | 288 | 252 |
| photographie | 6 | 829 | 558 | 220 | 51 | 37 |
| melange | 5 | 464 | 252 | 96 | 116 | 13 |

### Limites du jeu collecté
- **Likes masqués** : 567 posts (14 %) sans `like_count`, très inégalement répartis
  (29 % en ecriture contre 3 % en melange) — écartés de la cible en phase 2.
- **Vidéos** : 56 % du dataset, mais seule la miniature est disponible.
- **Historique plafonné** à 200 posts par compte : biais vers les publications récentes.
- **Type de post lié au thème** (art = 54 % de carrousels, photographie = 67 % de vidéos) :
  risque de confusion entre les deux variables.
- Business Discovery ne lit que les comptes Business/Creator publics ; les comptes
  rejetés sont listés dans `failed_accounts.csv` et ont été remplacés.

## Phase 2 : nettoyage et exploration

```
python -m src.processing.clean_data     # -> data/processed/dataset_clean.csv
python -m src.viz.build_report          # -> reports/exploration_summary.html
jupyter lab notebooks/01_exploration.ipynb
```

`clean_data.py` fusionne posts et profils, dérive les variables (`type_post`,
`likes_masques`, `interactions`, horodatage, comptages textuels) et répartit les comptes
en train / validation. Le jeu nettoyé n'est pas versionné : il contient le contenu de
comptes tiers et se régénère à partir des données brutes.

### La cible : `interactions`, et non un taux d'engagement

`interactions = likes + commentaires`

Le réflexe habituel serait le *taux d'engagement* (interactions ÷ abonnés). Il a été
écarté volontairement, parce que **le projet vise les petits comptes**. À cette échelle,
le taux devient instable : avec 48 abonnés, un seul like de plus déplace le taux de
2 points, contre 0,02 point pour un compte de 5 000. Diviser par un dénominateur
minuscule fabrique du bruit.

`followers` reste dans le jeu de données comme **variable explicative à part entière**,
jamais comme dénominateur. C'est au modèle d'apprendre son effet plutôt qu'à nous de
l'imposer par une division.

### Deux situations à ne pas confondre

| Cas | Ce que renvoie l'API | Décision | Volume |
|---|---|---|---|
| Le compte masque ses likes | clé `like_count` absente | écarté — cible inconnue | 567 |
| Le post a vraiment fait zéro | `like_count = 0` | conservé — un flop est un signal | 24 |

Confondre les deux reviendrait à fabriquer un total à partir des seuls commentaires pour
les comptes qui masquent leurs likes. Un post à zéro like peut d'ailleurs avoir des
commentaires, auquel cas ses `interactions` ne sont pas nulles.

### Découpage

**Le partage train / validation se fait par compte, jamais par post.** Deux posts d'un
même compte partagent son audience et son style : les répartir entre les deux groupes
ferait fuiter de l'information et gonflerait artificiellement le score.

| | |
|---|---|
| Posts exploitables | 3 587 sur 4 173 (86 %) |
| Train | 2 651 posts / 21 comptes |
| Validation | 936 posts / 7 comptes (26 %) |
| Cas d'usage | 19 posts / mon compte, jamais en entraînement |
| Cible médiane | 93 interactions |

### Une précaution de méthode : comparer chaque post à son compte

Avec `interactions` comme cible, un gros compte produit mécaniquement plus de réactions
qu'un petit. Comparer directement des posts de comptes différents reviendrait donc à
mesurer la taille des comptes. Les analyses utilisent donc une mesure relative :

`interactions_relatives = interactions ÷ médiane des interactions du compte`

Limite stricte : cette mesure vaut 1,0 pour tout groupe défini **au niveau du compte**
(thème, abonnés), puisque chaque compte est centré sur lui-même. Elle ne sert qu'aux
variables du **post**. Elle n'est pas enregistrée dans le jeu de données : calculée à
partir de la cible, elle constituerait une fuite si elle servait de variable d'entrée.

### Résultats

| Effet | Ampleur | Niveau |
|---|---|---|
| Taille du compte | rho = +0,37, plus lâche qu'attendu | compte |
| Thème du compte | rapport 2, confondu avec la taille | compte |
| Heure de publication | facteur 1,7, pic à 9 h UTC (+25 %) | post |
| Format du post | carrousel +5 %, image −6 % | post |
| Nombre de hashtags | 1 à 5 valent +9 %, au-delà l'effet s'annule | post |
| Présence d'emojis | +1 %, négligeable malgré p = 0,003 | post |
| Longueur de légende | rho = +0,08, sous 1 % de variation | post |
| Jour de la semaine | aucun (p = 0,86) | post |

Trois observations structurantes :

1. **La distribution est log-normale.** Médiane 93 interactions, moyenne 1 168, maximum
   528 383. La cible sera modélisée en `log(1 + interactions)`, le `+1` conservant les
   24 posts à zéro.
2. **La vidéo est un billet de loterie.** Médiane la plus basse des trois formats
   (77 contre 122 pour l'image), mais moyenne vingt fois supérieure à sa propre médiane.
3. **Les métadonnées textuelles simples n'expliquent presque rien.** L'essentiel de la
   variation reste inexpliqué — c'est ce que l'image et le sens de la légende devront
   capter en phase 3.

Le détail figure dans [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb)
(angle Data Scientist) et [`reports/exploration_summary.html`](reports/exploration_summary.html)
(angle Analyst, lecteur non technique).

### Ce que le changement de cible a modifié

L'exploration a d'abord été menée sur le taux d'engagement, avant de basculer sur
`interactions`. Trois conclusions ont changé au passage, et elles méritent d'être
consignées : ce sont elles qui justifient le choix de cible.

**1. L'effet de l'heure était surestimé d'un facteur 2,6.**
Première version : « publier à 15 h plutôt qu'à 21 h multiplie l'engagement par 4,5 ».
Après correction : **facteur 1,7, avec un pic à 9 h UTC**. L'écart initial venait d'un
taux calculé toutes publications confondues. Les créneaux horaires ne sont pas occupés
par les mêmes comptes, si bien que la mesure mélangeait l'effet de l'heure et la taille
des comptes qui publient à ces heures-là. Une fois chaque post comparé au sien, l'effet
retombe à 1,7.

**2. Le classement des formats s'inverse — un cas de paradoxe de Simpson.**
En valeurs brutes, l'image arrive en tête (122 interactions contre 77 pour la vidéo).
Une fois chaque post comparé à son propre compte, l'ordre change : **le carrousel passe
devant (+5 %) et l'image tombe dernière (−6 %)**. La première lecture reflétait surtout
le fait que les images proviennent de comptes plus actifs en interactions.

**3. Le lien entre taille de compte et résultat change de signe, et de lecture.**
Avec un taux, la corrélation était **négative** (rho = −0,39) : les petits comptes
engageaient proportionnellement plus. Avec `interactions`, elle devient **positive mais
lâche** (rho = +0,37, à peine hors du seuil de significativité sur 28 comptes). Les deux
résultats sont vrais et décrivent deux choses différentes. Le second est le plus utile
ici : un compte plus suivi récolte davantage, mais bien moins mécaniquement qu'on ne
l'imaginerait, **ce qui laisse une vraie marge au contenu** — précisément ce que le
projet cherche à exploiter pour les petits comptes.

Un effet secondaire, moins spectaculaire mais réel : l'analyse des tendances temporelles
devient honnête. Avec le taux, le dénominateur était le nombre d'abonnés *actuel*, si
bien que les vieux posts d'un compte ayant beaucoup grandi paraissaient artificiellement
faibles, créant une fausse progression. En interactions brutes, une hausse traduit une
vraie hausse des réactions.

### Biais connus de l'analyse
- **Heures en UTC**, pas dans le fuseau de chaque audience.
- **Associations, pas causalité** : rien ici ne prouve que publier à 9 h *provoque* plus
  de réactions.
- **Thèmes et taille de compte confondus** : les thèmes les mieux classés ne rassemblent
  pas des comptes de taille comparable.
- **Portée limitée** : l'échantillon couvre 255 à 11 209 abonnés dans des univers
  créatifs. Mon compte, avec 48 abonnés, se situe **sous la plage d'entraînement** : les
  prédictions le concernant resteront fragiles tant que le jeu ne comptera pas davantage
  de très petits comptes.
