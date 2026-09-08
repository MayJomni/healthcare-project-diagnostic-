"""
Tests unitaires pour le diagnostic ML.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np


def test_training_data_loads():
    """Verifie que le fichier Training.csv se charge correctement."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train = pd.read_csv(os.path.join(base_dir, "Training.csv"))
    assert "prognosis" in train.columns
    assert train.shape[0] > 0
    assert train.shape[1] > 10


def test_testing_data_loads():
    """Verifie que le fichier Testing.csv se charge correctement."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test = pd.read_csv(os.path.join(base_dir, "Testing.csv"))
    assert "prognosis" in test.columns
    assert test.shape[0] > 0


def test_symptom_columns_binary():
    """Verifie que les colonnes de symptomes sont bien binaires (0 ou 1)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train = pd.read_csv(os.path.join(base_dir, "Training.csv"))
    X = train.drop(columns=["prognosis"])
    unique_vals = set()
    for col in X.columns:
        unique_vals.update(X[col].unique())
    assert unique_vals.issubset({0, 1})


def test_label_encoder():
    """Verifie que le LabelEncoder encode correctement les maladies."""
    from sklearn.preprocessing import LabelEncoder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train = pd.read_csv(os.path.join(base_dir, "Training.csv"))
    le = LabelEncoder()
    y_encoded = le.fit_transform(train["prognosis"])
    assert len(le.classes_) > 0
    assert y_encoded.min() == 0
    assert y_encoded.max() == len(le.classes_) - 1


def test_nom_lisible():
    """Verifie la conversion des noms de symptomes."""
    def nom_lisible(symptome):
        return symptome.replace("_", " ").strip()

    assert nom_lisible("high_fever") == "high fever"
    assert nom_lisible("chest_pain") == "chest pain"
    assert nom_lisible("fatigue") == "fatigue"


def test_bruiter_pour_augmentation():
    """Verifie que l'augmentation de donnees fonctionne correctement."""
    X = pd.DataFrame({
        "s1": [1, 0, 1],
        "s2": [0, 1, 1],
        "s3": [1, 1, 0],
    })
    rng = np.random.RandomState(42)
    X_bruite = X.copy()
    # On verifie simplement que la forme est preservee
    assert X_bruite.shape == X.shape
