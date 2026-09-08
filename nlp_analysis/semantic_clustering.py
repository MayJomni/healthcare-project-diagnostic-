"""
Module pour le clustering sémantique de textes médicaux.
"""
import logging
from typing import List, Dict, Any

try:
    import numpy as np
except ImportError:
    logging.error("La bibliothèque 'numpy' est requise. Installez-la avec 'pip install numpy'.")
    raise

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logging.warning("La bibliothèque 'sentence-transformers' n'est pas installée. Utilisation du repli TF-IDF. Installez avec 'pip install sentence-transformers'.")

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    logging.error("La bibliothèque 'scikit-learn' est requise. Installez-la avec 'pip install scikit-learn'.")
    raise

try:
    from sklearn.manifold import TSNE
except ImportError:
    TSNE = None
    
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

class SemanticClusterer:
    """
    Classe pour effectuer le clustering sémantique sur un ensemble de textes.
    """
    def __init__(self):
        """
        Initialise le clusterer. Charge le modèle SentenceTransformer si disponible.
        """
        self.model = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logging.error(f"Erreur lors du chargement du modèle sentence-transformers : {e}")
                self.model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convertit une liste de textes en embeddings vectoriels.
        
        Args:
            texts (List[str]): La liste des textes à encoder.
            
        Returns:
            np.ndarray: Un tableau numpy contenant les embeddings.
        """
        if self.model is not None:
            return self.model.encode(texts)
        else:
            # Repli sur TF-IDF
            vectorizer = TfidfVectorizer(max_features=384) # pour avoir une taille similaire
            tfidf_matrix = vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray()

    def cluster(self, texts: List[str], n_clusters: int = 5) -> Dict[str, Any]:
        """
        Regroupe les textes en clusters en utilisant KMeans.
        
        Args:
            texts (List[str]): Les textes à clusteriser.
            n_clusters (int): Le nombre de clusters désiré.
            
        Returns:
            Dict[str, Any]: Dictionnaire contenant les informations des clusters.
        """
        if not texts:
            return {"clusters": [], "labels": [], "n_clusters": n_clusters}

        embeddings = self.embed_texts(texts)
        
        # Ajuster le nombre de clusters si moins de textes que de clusters
        n_clusters = min(n_clusters, len(texts))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        clusters_info = []
        for i in range(n_clusters):
            cluster_texts = [texts[j] for j in range(len(texts)) if labels[j] == i]
            
            # Déterminer un texte centroïde (le plus proche du centre)
            centroid = kmeans.cluster_centers_[i]
            cluster_embeddings = [embeddings[j] for j in range(len(texts)) if labels[j] == i]
            
            if cluster_embeddings:
                distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
                centroid_idx = np.argmin(distances)
                centroid_text = cluster_texts[centroid_idx]
            else:
                centroid_text = ""
                
            clusters_info.append({
                "cluster_id": i,
                "texts": cluster_texts,
                "centroid_text": centroid_text
            })
            
        return {
            "clusters": clusters_info,
            "labels": labels.tolist(),
            "n_clusters": n_clusters
        }

    def reduce_dimensions(self, embeddings: np.ndarray, method: str = 'tsne') -> np.ndarray:
        """
        Réduit les dimensions des embeddings pour la visualisation (t-SNE ou UMAP).
        
        Args:
            embeddings (np.ndarray): Les embeddings d'origine.
            method (str): La méthode de réduction ('tsne' ou 'umap').
            
        Returns:
            np.ndarray: Les embeddings réduits à 2 dimensions.
        """
        n_samples = embeddings.shape[0]
        # TSNE nécessite que perplexity soit inférieur au nombre d'échantillons
        perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

        if method.lower() == 'umap':
            if HAS_UMAP:
                reducer = umap.UMAP(n_components=2, random_state=42)
                return reducer.fit_transform(embeddings)
            else:
                logging.warning("UMAP n'est pas installé. Utilisation de t-SNE comme repli.")
                method = 'tsne'
                
        if method.lower() == 'tsne':
            if TSNE is None:
                raise ImportError("La bibliothèque scikit-learn est requise pour t-SNE.")
            if n_samples < 2:
                # Pas assez de points pour la réduction
                return np.zeros((n_samples, 2))
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            return reducer.fit_transform(embeddings)
            
        raise ValueError(f"Méthode de réduction de dimension non reconnue : {method}")
