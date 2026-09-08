import os
from dotenv import load_dotenv

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Clés API pour les modèles de langage
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Fournisseur LLM ('openai' ou 'anthropic')
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Chemins de base du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "historique.db")

# Indicateurs de fonctionnalités (Feature Flags)
ENABLE_LLM = os.getenv("ENABLE_LLM", "true").lower() == "true"
ENABLE_POSE = os.getenv("ENABLE_POSE", "true").lower() == "true"
ENABLE_NLP = os.getenv("ENABLE_NLP", "true").lower() == "true"
ENABLE_ECOSYSTEM = os.getenv("ENABLE_ECOSYSTEM", "true").lower() == "true"
