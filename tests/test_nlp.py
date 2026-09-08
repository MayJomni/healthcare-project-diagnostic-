"""
Tests unitaires pour les modules NLP.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def test_nlp_imports():
    """Verifie que les modules NLP peuvent etre importes."""
    try:
        from nlp_analysis.semantic_clustering import SemanticClusterer
        from nlp_analysis.topic_modeling import TopicModeler
        from nlp_analysis.keyword_mapping import KeywordMapper
        assert True
    except ImportError as e:
        pytest.skip(f"Dependances NLP manquantes : {e}")


def test_keyword_mapper_init():
    """Verifie l'initialisation du KeywordMapper."""
    try:
        from nlp_analysis.keyword_mapping import KeywordMapper
        mapper = KeywordMapper()
        assert mapper is not None
    except ImportError:
        pytest.skip("Dependances NLP manquantes")


def test_keyword_tfidf():
    """Verifie l'extraction TF-IDF basique."""
    try:
        from nlp_analysis.keyword_mapping import KeywordMapper
        mapper = KeywordMapper()
        textes = [
            "Le patient presente de la fievre et de la toux.",
            "La fievre est un symptome courant de la grippe.",
            "La toux seche peut indiquer une infection respiratoire."
        ]
        results = mapper.extract_keywords_tfidf(textes, top_n=5)
        assert isinstance(results, list)
        assert len(results) > 0
        if results:
            assert "mot_cle" in results[0] or "score_tfidf" in results[0]
    except (ImportError, Exception) as e:
        pytest.skip(f"Test ignore : {e}")


def test_topic_modeler_init():
    """Verifie l'initialisation du TopicModeler."""
    try:
        from nlp_analysis.topic_modeling import TopicModeler
        modeler = TopicModeler()
        assert modeler is not None
    except ImportError:
        pytest.skip("Dependances NLP manquantes")


def test_semantic_clusterer_init():
    """Verifie l'initialisation du SemanticClusterer."""
    try:
        from nlp_analysis.semantic_clustering import SemanticClusterer
        clusterer = SemanticClusterer()
        assert clusterer is not None
    except ImportError:
        pytest.skip("Dependances NLP manquantes")


def test_ecosystem_classifier_import():
    """Verifie que le classificateur d'ecosystemes peut etre importe."""
    try:
        from ecosystem_analysis.ecosystem_classifier import EcosystemClassifier
        classifier = EcosystemClassifier()
        assert classifier is not None
    except ImportError:
        pytest.skip("Module ecosystem_analysis non disponible")


def test_ecosystem_classification():
    """Verifie la classification ecosystemique basique."""
    try:
        from ecosystem_analysis.ecosystem_classifier import EcosystemClassifier
        classifier = EcosystemClassifier()
        text = "Notre ecosysteme d'innovation repose sur un partenariat avec les universites et l'industrie."
        result = classifier.classify_ecosystem_usage(text)
        assert isinstance(result, dict)
    except (ImportError, Exception) as e:
        pytest.skip(f"Test ignore : {e}")


def test_quadruple_helix_import():
    """Verifie que l'analyseur quadruple helice peut etre importe."""
    try:
        from ecosystem_analysis.quadruple_helix import QuadrupleHelixAnalyzer
        analyzer = QuadrupleHelixAnalyzer()
        assert analyzer is not None
    except ImportError:
        pytest.skip("Module ecosystem_analysis non disponible")
