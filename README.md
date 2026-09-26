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
`likes_masques`, `taux_engagement`, horodatage, comptages textuels) et répartit les
comptes en train / validation. Le jeu nettoyé n'est pas versionné : il contient le
contenu de comptes tiers et se régénère à partir des données brutes.

**Le découpage train / validation se fait par compte, jamais par post.** Deux posts
d'un même compte partagent son audience et son style : les répartir entre les deux
groupes ferait fuiter de l'information et gonflerait artificiellement le score.

### Résultat

| | |
|---|---|
| Posts exploitables | 3 587 sur 4 173 (86 %) |
| Train | 2 651 posts / 21 comptes |
| Validation | 936 posts / 7 comptes (26 %) |
| Cas d'usage | 19 posts / mon compte, jamais en entraînement |

| Effet sur l'engagement | Ampleur |
|---|---|
| Heure de publication | **Forte** — facteur 4,5 entre creux et pic |
| Taille du compte | **Forte** — rho = −0,39, sens inverse |
| Thème du compte | Modérée — rapport 3 entre extrêmes |
| Type de post | Modérée — image > carrousel > vidéo en médiane |
| Nombre de hashtags | Faible — 1 à 5 valent +9 %, au-delà l'effet s'annule |
| Longueur de légende | Très faible — environ 1 % de la variation |
| Présence d'emojis | Aucune — p = 0,25 |
| Jour de la semaine | Aucune — p = 0,40 |

Trois observations structurantes pour la suite :

1. **La distribution est log-normale** : médiane 3,3 % mais maximum 11 077 %. Quelques
   reels viraux dépassent 100 % d'engagement en touchant bien au-delà des abonnés.
   La cible sera modélisée en logarithme.
2. **La vidéo est un billet de loterie** : médiane la plus basse des trois formats
   (2,9 % contre 4,3 % pour l'image), mais moyenne quatre fois supérieure.
3. **Les métadonnées textuelles simples n'expliquent presque rien.** L'essentiel de la
   variation reste inexpliqué — c'est ce que l'image et le sens de la légende devront
   capter en phase 3.

Le détail figure dans [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb)
(angle Data Scientist) et [`reports/exploration_summary.html`](reports/exploration_summary.html)
(angle Analyst, lecteur non technique).

### Biais connus de l'analyse
- **Abonnés figés** : l'API ne donne le nombre d'abonnés qu'à la date de collecte, alors
  que les posts couvrent 2013-2026. Les publications anciennes des comptes ayant beaucoup
  grandi sont mécaniquement pénalisées, ce qui gonfle les tendances à la hausse.
- **Heures en UTC**, pas dans le fuseau de chaque audience.
- **Associations, pas causalité** : rien ici ne prouve que publier à 15 h *provoque* un
  meilleur engagement.
- **Thèmes et taille de compte confondus** : les thèmes les mieux classés rassemblent
  aussi le plus de petits comptes, qui engagent davantage.
