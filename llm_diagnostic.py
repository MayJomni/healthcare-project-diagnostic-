import json
import logging
from typing import Dict, List, Any
import config

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# Configuration du journal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMDiagnosticEngine:
    """
    Moteur de diagnostic basé sur les grands modèles de langage (LLM).
    Supporte OpenAI et Anthropic. Mode simulation si aucune clé n'est disponible.
    """

    def __init__(self) -> None:
        """
        Initialise le moteur de diagnostic en configurant le client approprié.
        """
        self.provider = config.LLM_PROVIDER
        self.openai_client = None
        self.anthropic_client = None
        self.mode_simulation = False

        if not config.OPENAI_API_KEY and not config.ANTHROPIC_API_KEY:
            logger.warning("Aucune clé API fournie. Mode simulation activé.")
            self.mode_simulation = True
        else:
            if self.provider == "openai" and config.OPENAI_API_KEY:
                if OpenAI:
                    self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
                else:
                    logger.error("La bibliothèque openai n'est pas installée.")
                    self.mode_simulation = True
            elif self.provider == "anthropic" and config.ANTHROPIC_API_KEY:
                if Anthropic:
                    self.anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
                else:
                    logger.error("La bibliothèque anthropic n'est pas installée.")
                    self.mode_simulation = True
            else:
                logger.warning("Clé API manquante pour le fournisseur choisi. Mode simulation activé.")
                self.mode_simulation = True

    def _obtenir_reponse_simulation(self, symptomes: List[str]) -> Dict[str, Any]:
        """
        Génère une réponse simulée en fonction des symptômes courants.
        
        :param symptomes: Liste des symptômes signalés.
        :return: Dictionnaire structuré contenant le diagnostic simulé.
        """
        symptomes_texte = " ".join(symptomes).lower()
        
        if "fièvre" in symptomes_texte and "toux" in symptomes_texte:
            return {
                "maladie_probable": "Grippe",
                "explication": "La présence de fièvre et de toux suggère une infection respiratoire virale telle que la grippe.",
                "niveau_confiance": 0.85,
                "recommandations": ["Se reposer", "Boire beaucoup d'eau", "Consulter un médecin si les symptômes s'aggravent."]
            }
        elif "mal de tête" in symptomes_texte and "nausée" in symptomes_texte:
            return {
                "maladie_probable": "Migraine",
                "explication": "Les maux de tête intenses accompagnés de nausées sont typiques des crises de migraine.",
                "niveau_confiance": 0.90,
                "recommandations": ["Se reposer dans une pièce sombre", "Prendre un analgésique", "Consulter si la douleur est persistante."]
            }
        elif "douleur thoracique" in symptomes_texte:
            return {
                "maladie_probable": "Problème cardiaque potentiel",
                "explication": "La douleur thoracique peut être le signe d'une affection grave comme une angine de poitrine ou un infarctus.",
                "niveau_confiance": 0.95,
                "recommandations": ["Appeler immédiatement les urgences (15 ou 112).", "Ne pas faire d'effort physique."]
            }
        elif "fatigue" in symptomes_texte and "soif" in symptomes_texte:
            return {
                "maladie_probable": "Diabète (possible)",
                "explication": "Une fatigue inexpliquée associée à une soif intense (polydipsie) peut indiquer un diabète.",
                "niveau_confiance": 0.70,
                "recommandations": ["Consulter un médecin", "Faire un test de glycémie à jeun."]
            }
        elif "éruption cutanée" in symptomes_texte and "démangeaison" in symptomes_texte:
            return {
                "maladie_probable": "Eczéma ou allergie cutanée",
                "explication": "Des éruptions cutanées prurigineuses (qui grattent) sont souvent le signe d'une dermatite ou d'une réaction allergique.",
                "niveau_confiance": 0.80,
                "recommandations": ["Appliquer une crème apaisante", "Éviter les allergènes connus", "Consulter un dermatologue."]
            }
        else:
            return {
                "maladie_probable": "Inconnu",
                "explication": "Les symptômes fournis ne correspondent pas de manière évidente à une condition courante dans notre base simulée.",
                "niveau_confiance": 0.40,
                "recommandations": ["Consulter un médecin pour un examen approfondi."]
            }

    def _construire_prompt(self, symptomes: List[str]) -> str:
        """
        Construit le prompt en français à envoyer au LLM.
        
        :param symptomes: Liste des symptômes.
        :return: Le texte du prompt.
        """
        liste_symptomes = ", ".join(symptomes)
        prompt = (
            f"En tant qu'assistant médical virtuel, veuillez analyser les symptômes suivants : {liste_symptomes}.\n\n"
            "Répondez UNIQUEMENT au format JSON avec les clés exactes suivantes :\n"
            "- maladie_probable (chaîne de caractères)\n"
            "- explication (chaîne de caractères)\n"
            "- niveau_confiance (nombre flottant entre 0 et 1)\n"
            "- recommandations (liste de chaînes de caractères)\n\n"
            "Assurez-vous que la réponse est un JSON valide sans aucun autre texte."
        )
        return prompt

    def diagnostiquer(self, symptomes: List[str]) -> Dict[str, Any]:
        """
        Évalue les symptômes et fournit un diagnostic structuré en interrogeant un LLM.
        
        :param symptomes: Liste des symptômes à analyser.
        :return: Dictionnaire contenant maladie_probable, explication, niveau_confiance et recommandations.
        """
        if not symptomes:
            return {
                "maladie_probable": "Erreur",
                "explication": "Aucun symptôme n'a été fourni.",
                "niveau_confiance": 0.0,
                "recommandations": ["Veuillez fournir une liste de symptômes."]
            }

        if self.mode_simulation:
            return self._obtenir_reponse_simulation(symptomes)

        prompt = self._construire_prompt(symptomes)

        try:
            if self.provider == "openai" and self.openai_client:
                reponse = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Vous êtes un expert médical qui renvoie des diagnostics précis au format JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                contenu = reponse.choices[0].message.content
                return json.loads(contenu)

            elif self.provider == "anthropic" and self.anthropic_client:
                reponse = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    system="Vous êtes un expert médical qui renvoie des diagnostics précis au format JSON. Ne renvoyez que le JSON.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                contenu = reponse.content[0].text
                return json.loads(contenu)
            else:
                return self._obtenir_reponse_simulation(symptomes)

        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON : {e}")
            return {
                "maladie_probable": "Erreur d'analyse",
                "explication": "Le modèle n'a pas renvoyé un JSON valide.",
                "niveau_confiance": 0.0,
                "recommandations": ["Réessayer avec des symptômes plus clairs."]
            }
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API du LLM : {e}")
            logger.warning("Basculement vers le mode simulation suite à une erreur.")
            return self._obtenir_reponse_simulation(symptomes)
