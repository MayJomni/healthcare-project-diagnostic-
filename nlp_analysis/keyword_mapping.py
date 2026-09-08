"""
Module pour l'extraction de mots-clés et l'analyse des cooccurrences.
"""
import json
import csv
import logging
from typing import List, Dict, Any

try:
    import numpy as np
except ImportError:
    logging.error("La bibliothèque 'numpy' est requise. Installez-la avec 'pip install numpy'.")
    raise

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
except ImportError:
    logging.error("La bibliothèque 'scikit-learn' est requise. Installez-la avec 'pip install scikit-learn'.")
    raise

try:
    from rake_nltk import Rake
    HAS_RAKE = True
except ImportError:
    HAS_RAKE = False
    logging.warning("La bibliothèque 'rake-nltk' n'est pas installée. Installez-la avec 'pip install rake-nltk'.")


class KeywordMapper:
    """
    Classe pour mapper et extraire des mots-clés depuis des textes.
    """
    def __init__(self):
        """
        Initialise l'extracteur de mots-clés.
        """
        self.french_stop_words = [
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'est', 
            'sont', 'dans', 'sur', 'pour', 'avec', 'par', 'ce', 'cette', 'ces',
            'il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous', 'au', 'aux',
            'qui', 'que', 'quoi', 'dont', 'où', 'ne', 'pas', 'plus', 'très', 'bien'
        ]
        
        if HAS_RAKE:
            try:
                # Tente de configurer RAKE pour le français
                self.rake = Rake(language='french')
            except Exception as e:
                logging.warning(f"Erreur lors de l'initialisation de RAKE pour le français: {e}")
                self.rake = None
        else:
            self.rake = None

    def extract_keywords_tfidf(self, texts: List[str], top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Extrait les mots-clés d'une liste de textes en utilisant TF-IDF.
        
        Args:
            texts (List[str]): Les documents textuels.
            top_n (int): Le nombre maximum de mots-clés à extraire.
            
        Returns:
            List[Dict[str, Any]]: Liste des mots-clés et de leurs scores TF-IDF globaux.
        """
        if not texts:
            return []
            
        vectorizer = TfidfVectorizer(stop_words=self.french_stop_words, max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return []
            
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculer le score moyen sur tous les documents
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Récupérer les top_n scores
        top_indices = mean_scores.argsort()[::-1][:top_n]
        
        keywords = []
        for idx in top_indices:
            if mean_scores[idx] > 0:
                keywords.append({
                    "mot_cle": feature_names[idx],
                    "score_tfidf": float(mean_scores[idx])
                })
                
        return keywords

    def extract_keywords_rake(self, text: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Extrait les mots-clés d'un texte unique en utilisant RAKE.
        
        Args:
            text (str): Le texte source.
            top_n (int): Le nombre maximum de mots-clés à extraire.
            
        Returns:
            List[Dict[str, Any]]: Liste des mots-clés avec leurs scores.
        """
        if not self.rake:
            logging.error("RAKE n'est pas disponible.")
            raise ImportError("Installez rake-nltk et configurez NLTK pour l'utiliser.")
            
        if not text:
            return []
            
        self.rake.extract_keywords_from_text(text)
        ranked_phrases = self.rake.get_ranked_phrases_with_scores()
        
        keywords = []
        for score, phrase in ranked_phrases[:top_n]:
            keywords.append({
                "mot_cle": phrase,
                "score": float(score)
            })
            
        return keywords

    def build_cooccurrence_map(self, texts: List[str], top_n: int = 30) -> Dict[str, Any]:
        """
        Construit une matrice de cooccurrence pour les mots les plus fréquents.
        
        Args:
            texts (List[str]): Les documents textuels.
            top_n (int): Nombre de mots les plus fréquents à conserver.
            
        Returns:
            Dict[str, Any]: Dictionnaire avec les mots, la matrice et les connexions.
        """
        if not texts:
            return {"mots": [], "matrice": [], "connexions": []}
            
        vectorizer = CountVectorizer(stop_words=self.french_stop_words, max_features=top_n)
        try:
            X = vectorizer.fit_transform(texts)
        except ValueError:
            return {"mots": [], "matrice": [], "connexions": []}
            
        words = vectorizer.get_feature_names_out().tolist()
        
        # Calculer la matrice de cooccurrence : X.T * X
        cooccurrence_matrix = (X.T * X)
        # Mettre la diagonale à zéro
        cooccurrence_matrix.setdiag(0)
        
        matrix_array = cooccurrence_matrix.toarray().tolist()
        
        connexions = []
        # Construire la liste des arêtes (connexions)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                force = matrix_array[i][j]
                if force > 0:
                    connexions.append({
                        "mot1": words[i],
                        "mot2": words[j],
                        "force": float(force)
                    })
                    
        # Trier les connexions par force décroissante
        connexions.sort(key=lambda x: x["force"], reverse=True)
        
        return {
            "mots": words,
            "matrice": matrix_array,
            "connexions": connexions
        }

    def export_results(self, results: Any, filepath: str, format: str = 'json') -> None:
        """
        Exporte les résultats dans un fichier JSON ou CSV.
        
        Args:
            results (Any): Les données à exporter.
            filepath (str): Le chemin du fichier de destination.
            format (str): Le format ('json' ou 'csv').
        """
        if format.lower() == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        elif format.lower() == 'csv':
            if isinstance(results, dict) and "connexions" in results:
                # Format spécifique pour les connexions de cooccurrence
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=["mot1", "mot2", "force"])
                    writer.writeheader()
                    writer.writerows(results["connexions"])
            elif isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
                # Format pour une liste de dictionnaires
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
            else:
                logging.error("Format de résultats non supporté pour l'export CSV.")
        else:
            raise ValueError(f"Format non supporté : {format}")
