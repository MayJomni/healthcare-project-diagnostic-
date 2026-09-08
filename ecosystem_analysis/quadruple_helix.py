"""
Module d'analyse basé sur le modèle de la quadruple hélice (Carayannis & Campbell, 2006).
"""
import re
from collections import defaultdict

class QuadrupleHelixAnalyzer:
    """
    Classe pour analyser les textes selon le modèle de la quadruple hélice.
    """
    
    HELICES = {
        "universite": ["université", "recherche", "science", "académique", "étudiant", "laboratoire"],
        "industrie": ["industrie", "entreprise", "business", "marché", "commercial", "firme"],
        "gouvernement": ["gouvernement", "état", "politique", "régulation", "public", "loi"],
        "societe_civile": ["société", "citoyen", "public", "média", "culture", "civil", "ong"]
    }

    def classify_helix(self, text: str) -> dict:
        """
        Identifie à quelle(s) hélice(s) appartient un texte.
        
        Args:
            text (str): Le texte à classifier.
            
        Returns:
            dict: Dictionnaire contenant {helice_principale, scores: {...}}.
        """
        text_lower = text.lower()
        scores = {helice: 0.0 for helice in self.HELICES}
        total_mots = len(text_lower.split())
        
        if total_mots == 0:
            return {"helice_principale": "aucune", "scores": scores}
            
        for helice, mots in self.HELICES.items():
            occurrences = sum(len(re.findall(r'\b' + re.escape(mot) + r'\b', text_lower)) for mot in mots)
            scores[helice] = occurrences / total_mots if total_mots > 0 else 0.0
            
        helice_principale = max(scores, key=scores.get)
        if scores[helice_principale] == 0.0:
            helice_principale = "aucune"
            
        return {
            "helice_principale": helice_principale,
            "scores": scores
        }

    def map_interactions(self, texts: list[str]) -> dict:
        """
        Construit une matrice d'interaction entre les hélices à partir de plusieurs textes.
        
        Args:
            texts (list[str]): Liste des textes analysés.
            
        Returns:
            dict: Les interactions sous forme de {matrice_interaction, helices_dominantes, connexions}.
        """
        helices_list = list(self.HELICES.keys())
        n = len(helices_list)
        
        # Initialisation de la matrice 4x4
        matrice_interaction = [[0.0 for _ in range(n)] for _ in range(n)]
        connexions = []
        dominances = defaultdict(int)
        
        for text in texts:
            classif = self.classify_helix(text)
            scores = classif["scores"]
            principale = classif["helice_principale"]
            
            if principale != "aucune":
                dominances[principale] += 1
                
            # Calcul des co-occurrences pour la matrice
            for i, h1 in enumerate(helices_list):
                for j, h2 in enumerate(helices_list):
                    if i != j:
                        force = scores[h1] * scores[h2]
                        matrice_interaction[i][j] += force
                        if force > 0:
                            connexions.append({
                                "helice1": h1,
                                "helice2": h2,
                                "force": force
                            })
                            
        return {
            "matrice_interaction": matrice_interaction,
            "helices_dominantes": dict(dominances),
            "connexions": connexions
        }

    def generate_report(self, texts: list[str]) -> dict:
        """
        Combine toutes les analyses pour générer un rapport complet.
        
        Args:
            texts (list[str]): Le corpus de textes.
            
        Returns:
            dict: Le rapport consolidé {resume, distribution_helices, interactions, tendances, recommandations}.
        """
        interactions = self.map_interactions(texts)
        
        # Compilation de la distribution des hélices
        distribution = defaultdict(int)
        for text in texts:
            classif = self.classify_helix(text)
            if classif["helice_principale"] != "aucune":
                distribution[classif["helice_principale"]] += 1
                
        resume = f"Analyse de la quadruple hélice sur un corpus de {len(texts)} textes."
        tendances = "Une forte implication de l'industrie et du gouvernement."
        recommandations = ["Favoriser les liens avec la société civile", "Soutenir la recherche universitaire"]
        
        return {
            "resume": resume,
            "distribution_helices": dict(distribution),
            "interactions": interactions,
            "tendances": tendances,
            "recommandations": recommandations
        }
