"""
Tests unitaires pour le module LLM.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def test_llm_engine_import():
    """Verifie que le moteur LLM peut etre importe."""
    from llm_diagnostic import LLMDiagnosticEngine
    assert LLMDiagnosticEngine is not None


def test_llm_engine_simulation_mode():
    """Verifie que le moteur LLM fonctionne en mode simulation."""
    from llm_diagnostic import LLMDiagnosticEngine
    engine = LLMDiagnosticEngine()
    # Sans cle API, le mode simulation doit etre actif
    assert engine.mode_simulation is True


def test_llm_diagnostiquer_empty():
    """Verifie la gestion des symptomes vides."""
    from llm_diagnostic import LLMDiagnosticEngine
    engine = LLMDiagnosticEngine()
    result = engine.diagnostiquer([])
    assert "maladie_probable" in result
    assert result["maladie_probable"] == "Erreur"


def test_llm_diagnostiquer_simulation():
    """Verifie que le diagnostic en simulation retourne un resultat structure."""
    from llm_diagnostic import LLMDiagnosticEngine
    engine = LLMDiagnosticEngine()
    result = engine.diagnostiquer(["fatigue", "toux"])
    assert "maladie_probable" in result
    assert "explication" in result
    assert "niveau_confiance" in result
    assert "recommandations" in result
    assert isinstance(result["recommandations"], list)


def test_llm_diagnostiquer_unknown_symptoms():
    """Verifie le comportement avec des symptomes inconnus."""
    from llm_diagnostic import LLMDiagnosticEngine
    engine = LLMDiagnosticEngine()
    result = engine.diagnostiquer(["xyz_inconnu", "abc_inconnu"])
    assert "maladie_probable" in result
    assert result["niveau_confiance"] is not None


def test_llm_prompt_construction():
    """Verifie que le prompt est correctement construit."""
    from llm_diagnostic import LLMDiagnosticEngine
    engine = LLMDiagnosticEngine()
    prompt = engine._construire_prompt(["fievre", "toux"])
    assert "fievre" in prompt
    assert "toux" in prompt
    assert "JSON" in prompt
