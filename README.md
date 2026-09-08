# Healthcare Diagnostic Platform

Plateforme de diagnostic medical et d'analyse de sante integrant IA, NLP et technologies de sante numerique.

## Modules

### 1. Diagnostic ML (4 modeles)
- **Random Forest**, **Decision Tree**, **Naive Bayes**, **XGBoost**
- Prediction de maladies a partir de 132 symptomes
- Consensus entre les 4 modeles
- Explicabilite : symptomes determinants

### 2. Diagnostic LLM (OpenAI / Anthropic)
- Analyse des symptomes par un grand modele de langage
- Support OpenAI (GPT-3.5) et Anthropic (Claude 3 Haiku)
- Mode simulation si aucune cle API n'est disponible

### 3. Analyse NLP
- **Clustering semantique** : regroupement de textes par similarite
- **Modelisation thematique** : extraction de sujets dominants (LDA / BERTopic)
- **Cartographie de mots-cles** : TF-IDF, RAKE, co-occurrences

### 4. Analyse d'ecosystemes
- Modele de la **quadruple helice** (Carayannis & Campbell, 2006)
- Classification des analogies ecosystemiques
- Identification des acteurs (dominants, challengers, facilitateurs)
- Scraping de rapports investisseurs et articles tech

### 5. Estimation de pose
- Detection posturale en temps reel (MediaPipe BlazePose)
- Analyse de posture et score de qualite
- Suivi d'exercices (squats, curls, extensions)
- Detection du risque de chute

### 6. Dashboard integre
- Interface unifiee avec navigation entre tous les modules
- Historique des diagnostics avec feedback utilisateur
- Analytics : correlation consensus/confiance (etude HCI)

## Installation

```bash
# Cloner le projet
git clone https://github.com/MayJomni/healthcare-project-diagnostic-.git
cd healthcare-project-diagnostic-

# Installer les dependances
pip install -r requirements.txt

# Configurer les variables d'environnement (optionnel, pour le module LLM)
cp .env.example .env
# Editer .env avec vos cles API

# Lancer l'application
python app.py
```

L'application sera accessible sur http://localhost:5000

## Routes principales

| Route | Description |
|-------|-------------|
| `/` | Page de diagnostic (selection de symptomes) |
| `/dashboard` | Tableau de bord principal |
| `/nlp` | Interface d'analyse NLP |
| `/pose` | Estimation de pose en temps reel |
| `/ecosystem` | Analyse d'ecosystemes |
| `/historique` | Historique des diagnostics |
| `/analytics` | Analytics du feedback |

## API Endpoints

| Endpoint | Methode | Description |
|----------|---------|-------------|
| `/predict` | POST | Diagnostic ML (4 modeles) |
| `/predict_llm` | POST | Diagnostic LLM |
| `/feedback` | POST | Soumission de feedback |
| `/api/nlp/cluster` | POST | Clustering semantique |
| `/api/nlp/topics` | POST | Modelisation thematique |
| `/api/nlp/keywords` | POST | Extraction de mots-cles |
| `/api/ecosystem/analyze` | POST | Analyse d'ecosysteme |
| `/api/ecosystem/scrape` | POST | Scraping d'articles |
| `/video_feed` | GET | Flux video avec pose |
| `/api/pose/status` | GET | Statut du module pose |
| `/api/status` | GET | Statut de tous les modules |

## Structure du projet

```
healthcare-project-diagnostic-/
├── app.py                    # Application Flask principale
├── config.py                 # Configuration centralisee
├── llm_diagnostic.py         # Moteur de diagnostic LLM
├── requirements.txt          # Dependances Python
├── .env.example              # Template variables d'environnement
├── README.md                 # Ce fichier
│
├── nlp_analysis/             # Module NLP
│   ├── __init__.py
│   ├── semantic_clustering.py
│   ├── topic_modeling.py
│   └── keyword_mapping.py
│
├── ecosystem_analysis/       # Module ecosystemes
│   ├── __init__.py
│   ├── report_scraper.py
│   ├── ecosystem_classifier.py
│   └── quadruple_helix.py
│
├── pose_estimation/          # Module estimation de pose
│   ├── __init__.py
│   ├── pose_detector.py
│   ├── health_analyzer.py
│   └── video_stream.py
│
├── tests/                    # Tests unitaires
│   ├── test_diagnostic.py
│   ├── test_llm.py
│   ├── test_nlp.py
│   └── test_api.py
│
├── templates/                # Templates HTML
│   ├── index.html
│   ├── historique.html
│   ├── dashboard.html
│   ├── nlp_dashboard.html
│   ├── pose_view.html
│   └── ecosystem_report.html
│
├── static/                   # Fichiers statiques
│   ├── script.js
│   ├── feedback.js
│   └── style.css
│
├── Training.csv              # Donnees d'entrainement
└── Testing.csv               # Donnees de test
```

## Technologies utilisees

- **Backend** : Flask, Python 3.x
- **ML** : scikit-learn, XGBoost
- **LLM** : OpenAI API, Anthropic API
- **NLP** : sentence-transformers, gensim, BERTopic, NLTK
- **Vision** : MediaPipe, OpenCV, TensorFlow
- **Frontend** : HTML5, CSS3, JavaScript vanilla
- **Base de donnees** : SQLite

## Competences couvertes

- Python avance et developpement logiciel
- Intelligence artificielle et apprentissage automatique
- Traitement du langage naturel (NLP)
- Integration d'API LLM (OpenAI, Anthropic)
- Traitement video en temps reel et estimation de pose
- Analyse de donnees quantitatives et qualitatives
- Integration front-end / back-end
- Interaction homme-machine (HCI)
- Tests logiciels et revue de code
- Analyse d'ecosystemes et modele quadruple helice
