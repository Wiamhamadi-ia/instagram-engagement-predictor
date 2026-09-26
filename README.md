# Prédiction d'engagement Instagram

Prédire l'engagement d'un post Instagram avant publication (image + légende + heure), avec CV, NLP et séries temporelles, puis une app Streamlit.

## Phases
1. **Collecte** (en cours) : API Instagram Graph (Business Discovery), 26 comptes publics + mon compte
2. Exploration & nettoyage
3. Feature engineering (CV, NLP, temporel)
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

### Limites à traiter en phase 2
- **Likes masqués** : 567 posts (14 %) sans `like_count`, très inégalement répartis
  (29 % en ecriture contre 3 % en melange) — exclure de la cible ou traiter à part.
- **Vidéos** : 56 % du dataset, mais seule la miniature est disponible.
- **Historique plafonné** à 200 posts par compte : biais vers les publications récentes.
- **Type de post lié au thème** (art = 54 % de carrousels, photographie = 67 % de vidéos) :
  risque de confusion entre les deux variables.
- Business Discovery ne lit que les comptes Business/Creator publics ; les comptes
  rejetés sont listés dans `failed_accounts.csv` et ont été remplacés.
