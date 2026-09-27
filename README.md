# Prédiction d'engagement Instagram

Prédire l'engagement d'un post Instagram avant publication (image + légende + heure), avec CV, NLP et séries temporelles, puis une app Streamlit.

## Phases
1. **Collecte** : API Instagram Graph (Business Discovery), 28 comptes publics + mon compte
2. **Exploration & nettoyage** : dataset propre, notebook d'analyse, rapport visuel
3. **Feature engineering** : embeddings image (CLIP) et texte (multilingue), temporel
4. **Modélisation** : baselines, trois candidats, ablation, diagnostic
5. App Streamlit — *en cours*
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

## Phase 3 : feature engineering

```
python -m src.features.build_features            # réutilise le cache d'embeddings
python -m src.features.build_features --recalc   # recalcule tout (~15 min)
```

Produit `data/processed/features.csv` : **4 173 lignes × 917 colonnes**, une ligne par
post, jointure sur `id`.

| Bloc | Colonnes | Contenu |
|---|---|---|
| Identifiant et cible | 4 | `id`, `log_interactions`, `groupe_split`, `utilisable_entrainement` |
| Temporel | 3 | `heure_sin`, `heure_cos`, `jours_depuis_dernier_post` |
| Texte | 2 | `hashtags_tranche` (0 à 4), `caption_vide` |
| Compte et format | 9 | `followers`, `type_*` (3), `theme_*` (5) |
| Standardisé | 2 | `followers_z`, `jours_depuis_dernier_post_z` |
| Image | 1 | `image_manquante` |
| Embeddings NLP | 384 | `nlp_000` … `nlp_383` |
| Embeddings CLIP | 512 | `cv_000` … `cv_511` |

La cible est `log_interactions = log(1 + interactions)`, renseignée uniquement pour les
3 587 posts entraînables. Le `+1` conserve les 24 posts à zéro interaction.

### Choix guidés par la phase 2

Le feature engineering applique directement ce que l'exploration a mesuré, plutôt que
d'empiler des variables par réflexe :

- **Heure encodée en sinus/cosinus**, jamais en valeur brute 0-23 : sans cela, 23 h et
  0 h seraient les deux points les plus éloignés alors qu'ils se suivent.
- **Jour de la semaine absent** : aucun effet mesuré en phase 2 (p = 0,86).
- **Hashtags en tranches ordinales** (0, 1-5, 6-10, 11-20, 21+) : l'effet est en cloche,
  une variable linéaire le manquerait entièrement.
- **Ni longueur de légende, ni nombre de mots, ni emojis** : testés et non pertinents,
  tous sous 1 % de variation expliquée.
- **`type_post` conservé en plus des embeddings image** : CLIP voit une miniature de
  vidéo comme une image fixe et ne peut pas deviner le format.

### Le modèle de texte a dû être changé

La consigne initiale suggérait `all-MiniLM-L6-v2`. Un test préalable l'a écarté :
**28,7 % des légendes exploitables sont en écriture non latine**, dont 996 en arabe, et
le thème *ecriture* est arabe à 97 %.

Sur quatre phrases arabes — deux sur l'amour, deux sur la mer — le pouvoir de séparation
mesuré, écart entre similarité intra-sujet et inter-sujet, donne :

| Modèle | Pouvoir de séparation |
|---|---|
| `all-MiniLM-L6-v2` (anglophone) | **−0,04** — juge « amour » et « mer » plus proches que deux phrases synonymes |
| `paraphrase-multilingual-MiniLM-L12-v2` | **+0,60** |

Le modèle anglophone aurait produit du bruit pur pour près d'un tiers du jeu de données,
en éliminant silencieusement tout un thème. La variante multilingue sort la même
dimension (384) pour un coût comparable.

Vérification après coup sur le jeu réel : les légendes arabes du thème *ecriture* ont une
similarité interne de 0,528, contre 0,341 face aux légendes latines du thème *nature*.
Côté image, chaque thème a une cohérence interne supérieure à l'externe (écart de +0,05
pour *nature* à +0,18 pour *ecriture*).

### Aucune fuite de données

- **Les embeddings sont calculés post par post**, indépendamment les uns des autres :
  ni moyenne, ni ajustement global, donc aucune circulation d'information entre posts.
- **La normalisation L2 des vecteurs est faite ligne par ligne**, chaque vecteur
  indépendamment.
- **La standardisation de `followers` et `jours_depuis_dernier_post` utilise uniquement
  les 2 651 posts de train.** Les paramètres sont enregistrés dans `scaler_params.json`
  et appliqués tels quels à la validation. Vérification : la moyenne de `followers_z`
  vaut exactement 0 sur le train et −0,53 sur la validation, ce qui prouve que les
  statistiques ne viennent pas du jeu complet.
- **La mesure relative au compte de la phase 2 n'est volontairement pas une variable** :
  calculée à partir de la cible, elle constituerait une fuite directe.

### Mise en cache

Les embeddings coûtent environ 15 minutes de calcul sur processeur. Ils sont conservés
dans `embeddings_nlp.npy` et `embeddings_cv.npy`, **indexés par identifiant de post**
plutôt que par position. Une relance complète prend alors 4 secondes, et ajouter des
comptes plus tard ne recalculera que les posts nouveaux.

### Limites du jeu de variables
- **Les vidéos ne sont vues que par leur miniature.** 56 % du jeu est constitué de
  reels, dont le mouvement, le rythme et le son échappent totalement au modèle.
- **Les carrousels ne sont vus que par leur première image**, alors qu'ils comptent
  jusqu'à dix visuels.
- **CLIP n'a pas été réentraîné** sur ce domaine : ses vecteurs décrivent le contenu
  visuel générique, pas ce qui fait le style d'un compte.

## Phase 4 : modélisation

```
python -m src.models.train_models        # ~25 min, produit resultats_modeles.json
jupyter lab notebooks/02_modelisation.ipynb
```

### Résultat principal : aucun modèle ne bat la baseline

| Modèle | RMSE (log) | Facteur d'erreur |
|---|---|---|
| Baseline : médiane du train | 1,402 | ×3,13 |
| **Baseline : régression sur `followers` seul** | **1,362** | **×3,03** |
| RandomForest (5 graines) | 1,424 ± 0,005 | ×3,25 |
| XGBoost (5 graines) | 1,419 ± 0,010 | ×3,20 |
| LightGBM (5 graines) | 1,404 ± 0,015 | ×3,13 |
| LightGBM réglé, 911 variables | 1,368 | ×3,03 |
| LightGBM, tabulaire + texte seulement | 1,365 | ×3,02 |
| LightGBM restreint aux comptes < 2 000 abonnés | 1,472 | ×3,33 |

**Trois algorithmes, 911 variables, un réglage par recherche aléatoire : le meilleur
score (1,365) reste au niveau d'une droite ajustée sur le seul nombre d'abonnés
(1,362).** C'est le résultat de la phase, et il est rapporté tel quel.

L'erreur typique correspond à un **facteur 3** : un post qui obtient 100 réactions est
prédit entre 33 et 300. Le nuage predit/observé du notebook montre le symptôme —
les prédictions restent comprises entre 4 et 5 en échelle log alors que la réalité
s'étale de 0 à 12. Le modèle prédit presque toujours la même chose.

### Protocole

- **Découpage par compte**, jamais par post.
- **Le réglage n'utilise jamais la validation** : recherche aléatoire de 18
  combinaisons, évaluées par `GroupKFold` groupé par compte *à l'intérieur du train*.
  Régler sur la validation puis y mesurer la performance donnerait un score optimiste.
- **Cinq graines par modèle**, pour distinguer un vrai écart d'une fluctuation.
- **Erreur détaillée par compte de validation** : aucun compte ne domine. Le maximum
  est 19,1 % de l'erreur totale pour une répartition égale à 14,3 %, donc le score
  agrégé n'est pas l'artefact d'un compte aberrant.

### Ablation : d'où vient le signal

| Bloc | Variables | RMSE |
|---|---|---|
| Tabulaire seul | 15 | 1,533 |
| Tabulaire + image | 527 | 1,419 |
| Tout | 911 | 1,368 |
| Tabulaire + texte | 399 | **1,365** |

Les 15 variables tabulaires seules font **pire que la médiane** (1,533 contre 1,402).
Les embeddings de texte sont le seul apport net. Les embeddings d'image n'aident pas et
dégradent légèrement le score quand on les ajoute au texte — cohérent avec le fait que
56 % du jeu est constitué de reels dont on ne voit qu'une miniature figée.

### Petits comptes : plus de données l'emporte

Évalués sur le même sous-ensemble de validation (360 posts, comptes < 2 000 abonnés) :

| Entraînement | Posts | Comptes | RMSE |
|---|---|---|---|
| Échantillon complet | 2 651 | 21 | **1,348** |
| Comptes < 2 000 abonnés | 912 | 6 | 1,472 |

Le modèle entraîné sur tout généralise mieux, y compris sur le terrain des petits
comptes : la spécialisation ne compense pas la perte de volume. **Conclusion fragile
cependant**, la validation ne comptant que 3 comptes sous ce seuil.

### L'anomalie de l'heure, et sa résolution

Le contrôle de cohérence avec la phase 2 a signalé une anomalie : `heure_sin` et
`heure_cos` ont une importance quasi nulle (0,0003) alors que la phase 2 y mesurait un
effet net, facteur 1,7 avec p < 0,001.

La décomposition de la variance donne la réponse : **l'heure explique 0,58 % de la
variance totale** à prédire, soit 1,0 % de la variance interne aux comptes. Les deux
résultats sont exacts mais ne parlent pas de la même échelle :

- La **phase 2** mesurait un décalage de médiane *à l'intérieur* de chaque compte. Sur
  cette échelle relative, +25 % est visible et actionnable pour un créateur.
- Le **modèle** prédit une valeur absolue, dont 44 % de la variance vient de l'écart
  *entre* comptes. À cette échelle, un effet à 0,58 % est noyé.

Ce n'est pas un bug mais une leçon : **un effet peut être statistiquement solide,
pratiquement utile, et malgré tout négligeable pour un modèle de prédiction.**

### Importance native contre importance par permutation

L'importance native (gain) attribue 40 % aux images et 33 % au texte. L'importance par
permutation, mesurée sur la validation, raconte l'inverse :

| Bloc | Dégradation du RMSE quand on le mélange |
|---|---|
| image (embeddings) | +0,024 |
| theme | +0,014 |
| heure | +0,000 |
| hashtags | −0,000 |
| format | −0,004 |
| texte (embeddings) | −0,012 |
| followers | −0,012 |
| jours depuis le dernier post | −0,012 |

Une valeur **négative** signifie que mélanger le bloc *améliore* la prédiction : le
modèle s'appuie dessus d'une façon qui ne se transpose pas à des comptes jamais vus.
L'importance native ne mesure que l'usage fait des variables pendant l'entraînement ;
avec 896 dimensions disponibles, un arbre trouve toujours où découper, même dans du
bruit.

### Pourquoi ça ne marche pas, et ce qu'il faut changer

**Le problème est le nombre de comptes, pas le nombre de posts.** Le modèle dispose de
2 651 exemples mais de seulement **21 comptes** d'entraînement, alors que 44 % de la
variance à prédire se situe *entre* les comptes. Apprendre à généraliser d'un compte à
l'autre sur 21 exemples n'est pas réaliste.

Trois conséquences pour la suite :

1. **Collecter plus de comptes prime sur tout le reste.** Doubler le nombre de posts par
   compte n'aiderait pas ; passer de 21 à 60 comptes, si.
2. **L'application de la phase 5 doit annoncer son incertitude** plutôt qu'afficher un
   nombre unique. Une fourchette et un rang relatif sont honnêtes ; « ce post fera
   240 réactions » ne l'est pas.
3. **Une cible relative au compte serait plus réaliste.** Prédire « ce post fera mieux
   ou moins bien que l'ordinaire de ce compte » élimine les 44 % de variance
   inter-comptes que le modèle ne sait pas traiter, et correspond mieux à la question
   que se pose un créateur.
