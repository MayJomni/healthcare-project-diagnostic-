"""
Module pour la modélisation de sujets (Topic Modeling) sur des textes médicaux.
"""
import logging
from typing import List, Dict, Any

try:
    import numpy as np
except ImportError:
    logging.error("La bibliothèque 'numpy' est requise. Installez-la avec 'pip install numpy'.")
    raise

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
except ImportError:
    logging.error("La bibliothèque 'scikit-learn' est requise. Installez-la avec 'pip install scikit-learn'.")
    raise

try:
    from bertopic import BERTopic
    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False
    logging.info("BERTopic non installé. Installez-le avec 'pip install bertopic' pour des fonctionnalités avancées.")


class TopicModeler:
    """
    Classe pour extraire et analyser les sujets dans un corpus de textes.
    """
    
    def __init__(self):
        """
        Initialise le modélisateur de sujets.
        """
        # Mots vides (stop words) en français pour sklearn
        self.french_stop_words = [
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'est', 
            'sont', 'dans', 'sur', 'pour', 'avec', 'par', 'ce', 'cette', 'ces',
            'il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous', 'au', 'aux',
            'qui', 'que', 'quoi', 'dont', 'où', 'ne', 'pas', 'plus', 'très', 'bien'
        ]
        self.vectorizer = None
        self.lda_model = None

    def extract_topics(self, texts: List[str], n_topics: int = 5) -> List[Dict[str, Any]]:
        """
        Extrait les sujets d'une liste de textes en utilisant LDA.
        
        Args:
            texts (List[str]): Les textes à analyser.
            n_topics (int): Le nombre de sujets à extraire.
            
        Returns:
            List[Dict[str, Any]]: Une liste de dictionnaires représentant les sujets trouvés.
        """
        if not texts:
            return []
            
        n_topics = min(n_topics, len(texts))
        
        # Vectorisation
        self.vectorizer = CountVectorizer(stop_words=self.french_stop_words, max_df=0.95, min_df=1)
        try:
            dtm = self.vectorizer.fit_transform(texts)
        except ValueError as e:
            logging.error(f"Erreur lors de la vectorisation : {e}")
            return []
            
        # Application de LDA
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.lda_model.fit(dtm)
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            # Trier les mots par leur poids
            top_features_ind = topic.argsort()[:-10 - 1:-1]
            mots_cles = [feature_names[i] for i in top_features_ind]
            poids = [float(topic[i]) for i in top_features_ind]
            
            topics.append({
                "topic_id": topic_idx,
                "mots_cles": mots_cles,
                "poids": poids
            })
            
        return topics

    def get_document_topics(self, text: str) -> List[Dict[str, Any]]:
        """
        Obtient la distribution des sujets pour un document donné.
        
        Args:
            text (str): Le texte à analyser.
            
        Returns:
            List[Dict[str, Any]]: La distribution des probabilités sur les sujets.
        """
        if not self.lda_model or not self.vectorizer:
            raise ValueError("Le modèle LDA n'a pas été entraîné. Appelez 'extract_topics' d'abord.")
            
        dtm = self.vectorizer.transform([text])
        topic_distribution = self.lda_model.transform(dtm)[0]
        
        distribution = []
        for topic_idx, prob in enumerate(topic_distribution):
            distribution.append({
                "topic_id": topic_idx,
                "probabilite": float(prob)
            })
            
        # Trier par probabilité décroissante
        distribution.sort(key=lambda x: x["probabilite"], reverse=True)
        return distribution
        
    def extract_topics_bertopic(self, texts: List[str]) -> Dict[str, Any]:
        """
        Extrait les sujets en utilisant BERTopic si disponible.
        
        Args:
            texts (List[str]): Les textes à analyser.
            
        Returns:
            Dict[str, Any]: Informations sur les sujets générés par BERTopic.
        """
        if not HAS_BERTOPIC:
            raise ImportError("BERTopic n'est pas installé. Utilisez extract_topics avec LDA.")
            
        model = BERTopic(language="french")
        topics, probabilities = model.fit_transform(texts)
        
        topic_info = model.get_topic_info()
        
        return {
            "topics": topics,
            "topic_info": topic_info.to_dict(orient="records")
        }
