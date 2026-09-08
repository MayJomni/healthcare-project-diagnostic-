"""
Tests unitaires pour les routes API Flask.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json


@pytest.fixture
def client():
    """Cree un client de test Flask."""
    # Desactiver les modules lourds pour les tests
    os.environ["ENABLE_POSE"] = "false"
    os.environ["ENABLE_NLP"] = "false"
    os.environ["ENABLE_ECOSYSTEM"] = "false"
    os.environ["ENABLE_LLM"] = "true"

    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page(client):
    """Verifie que la page d'accueil se charge."""
    response = client.get("/")
    assert response.status_code == 200


def test_dashboard_page(client):
    """Verifie que le dashboard se charge."""
    response = client.get("/dashboard")
    assert response.status_code == 200


def test_predict_no_symptoms(client):
    """Verifie l'erreur si aucun symptome n'est fourni."""
    response = client.post(
        "/predict",
        data=json.dumps({"symptomes": []}),
        content_type="application/json"
    )
    assert response.status_code == 400


def test_predict_with_symptoms(client):
    """Verifie la prediction avec des symptomes valides."""
    response = client.post(
        "/predict",
        data=json.dumps({"symptomes": ["high_fever", "cough"]}),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "resultats" in data
    assert "consensus" in data
    assert len(data["resultats"]) == 4  # 4 modeles ML


def test_predict_llm_with_symptoms(client):
    """Verifie le diagnostic LLM (mode simulation)."""
    response = client.post(
        "/predict_llm",
        data=json.dumps({"symptomes": ["fatigue", "cough"]}),
        content_type="application/json"
    )
    if response.status_code == 503:
        pytest.skip("Module LLM non disponible")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "resultat" in data
    assert "modele" in data


def test_feedback_missing_id(client):
    """Verifie l'erreur si historique_id est manquant."""
    response = client.post(
        "/feedback",
        data=json.dumps({"trust_score": 3}),
        content_type="application/json"
    )
    assert response.status_code == 400


def test_feedback_invalid_score(client):
    """Verifie l'erreur si trust_score est invalide."""
    response = client.post(
        "/feedback",
        data=json.dumps({"historique_id": 1, "trust_score": "abc"}),
        content_type="application/json"
    )
    assert response.status_code == 400


def test_historique_page(client):
    """Verifie que la page historique se charge."""
    response = client.get("/historique")
    assert response.status_code == 200


def test_analytics_page(client):
    """Verifie que la page analytics retourne du JSON."""
    response = client.get("/analytics")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "nb_reponses" in data


def test_api_status(client):
    """Verifie la route de statut des modules."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "modules" in data
    assert "diagnostic_ml" in data["modules"]
    assert data["modules"]["diagnostic_ml"] is True


def test_nlp_disabled(client):
    """Verifie que les routes NLP retournent 503 quand desactivees."""
    response = client.post(
        "/api/nlp/cluster",
        data=json.dumps({"textes": ["test1", "test2"]}),
        content_type="application/json"
    )
    assert response.status_code == 503


def test_ecosystem_disabled(client):
    """Verifie que les routes ecosystemes retournent 503 quand desactivees."""
    response = client.post(
        "/api/ecosystem/analyze",
        data=json.dumps({"texte": "test"}),
        content_type="application/json"
    )
    assert response.status_code == 503
