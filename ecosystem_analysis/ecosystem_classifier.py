"""
Module pour classifier l'utilisation du concept d'écosystème.
"""
import re
from collections import defaultdict

class EcosystemClassifier:
    """
    Classe pour classifier et analyser l'analogie de l'écosystème dans les textes d'affaires.
    """
    
    ECOSYSTEM_KEYWORDS = {
        "innovation": ["innovation", "recherche", "développement", "r&d", "brevet"],
        "plateforme": ["plateforme", "platform", "infrastructure", "api", "réseau"],
        "partenariat": ["partenariat", "alliance", "collaboration", "joint-venture"],
        "coopetition": ["coopétition", "coopetition", "concurrent", "coopération"],
        "disruption": ["disruption", "perturbation", "transformation", "changement"],
        "valeur_ajoutee": ["valeur ajoutée", "création de valeur", "bénéfice"],
        "parties_prenantes": ["parties prenantes", "stakeholders", "acteurs", "communauté"]
    }

    ACTOR_KEYWORDS = {
        "dominant": ["leader", "monopole", "géant", "dominant"],
        "challenger": ["challenger", "concurrent", "start-up", "nouveau venu"],
        "facilitateur": ["facilitateur", "enabler", "soutien", "catalyseur"],
        "intermediaire": ["intermédiaire", "courtier", "plateforme tierce"]
    }

    def classify_ecosystem_usage(self, text: str) -> dict:
        """
        Détecte les mots-clés liés à l'écosystème et classifie le type d'analogie utilisé.
        
        Args:
            text (str): Le texte à analyser.
            
        Returns:
            dict: Les résultats de la classification {type_analogie, mots_detectes, frequence, contexte_phrases}.
        """
        text_lower = text.lower()
        mots_detectes = []
        frequence = defaultdict(int)
        scores = defaultdict(int)
        
        # Détection des mots-clés par catégorie
        for categorie, mots in self.ECOSYSTEM_KEYWORDS.items():
            for mot in mots:
                occurrences = len(re.findall(r'\b' + re.escape(mot) + r'\b', text_lower))
                if occurrences > 0:
                    mots_detectes.append(mot)
                    frequence[mot] += occurrences
                    scores[categorie] += occurrences
                    
        type_analogie = max(scores, key=scores.get) if scores else "inconnu"
        
        # Extraction du contexte (phrases contenant les mots détectés)
        phrases = [p.strip() for p in re.split(r'[.!?]', text) if p.strip()]
        contexte_phrases = []
        for phrase in phrases:
            if any(mot in phrase.lower() for mot in mots_detectes):
                contexte_phrases.append(phrase)

        return {
            "type_analogie": type_analogie,
            "mots_detectes": mots_detectes,
            "frequence": dict(frequence),
            "contexte_phrases": contexte_phrases
        }

    def identify_actors(self, text: str) -> dict:
        """
        Identifie les types d'acteurs (dominant, challenger, facilitateur, intermediaire).
        
        Args:
            text (str): Le texte à analyser.
            
        Returns:
            dict: Les acteurs identifiés {acteurs: list[{nom, type, mentions}]}.
        """
        text_lower = text.lower()
        acteurs = []
        
        for type_acteur, mots in self.ACTOR_KEYWORDS.items():
            mentions = 0
            for mot in mots:
                mentions += len(re.findall(r'\b' + re.escape(mot) + r'\b', text_lower))
                
            if mentions > 0:
                acteurs.append({
                    "nom": type_acteur,  # Simplification, dans la réalité on extrairait le vrai nom
                    "type": type_acteur,
                    "mentions": mentions
                })
                
        return {"acteurs": acteurs}

    def analyze_corpus(self, texts: list[str]) -> dict:
        """
        Analyse un ensemble de textes.
        
        Args:
            texts (list[str]): Liste de textes à analyser.
            
        Returns:
            dict: Statistiques globales agrégées.
        """
        stats = {
            "total_textes": len(texts),
            "repartition_types": defaultdict(int),
            "mots_les_plus_frequents": defaultdict(int),
            "acteurs_detectes": defaultdict(int)
        }
        
        for text in texts:
            # Classification
            classif = self.classify_ecosystem_usage(text)
            stats["repartition_types"][classif["type_analogie"]] += 1
            
            for mot, freq in classif["frequence"].items():
                stats["mots_les_plus_frequents"][mot] += freq
                
            # Acteurs
            acteurs = self.identify_actors(text)
            for acteur in acteurs["acteurs"]:
                stats["acteurs_detectes"][acteur["type"]] += 1
                
        # Conversion des defaultdict en dict pour le retour final
        stats["repartition_types"] = dict(stats["repartition_types"])
        stats["mots_les_plus_frequents"] = dict(stats["mots_les_plus_frequents"])
        stats["acteurs_detectes"] = dict(stats["acteurs_detectes"])
        
        return stats
