"""
Etape 2 : Preparer X (symptomes) et y (maladie a predire)
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Charger les donnees (deja nettoyees a l'etape 1)
train = pd.read_csv("Training.csv")
test = pd.read_csv("Testing.csv")

# X = toutes les colonnes SAUF prognosis (ce sont les symptomes)
X_train = train.drop(columns=["prognosis"])
y_train = train["prognosis"]

X_test = test.drop(columns=["prognosis"])
y_test = test["prognosis"]

# Verification : X doit avoir 132 colonnes (symptomes), y doit etre une seule colonne
print("X_train shape:", X_train.shape)   # (4920, 132)
print("y_train shape:", y_train.shape)   # (4920,)

# Regarder a quoi ressemble une ligne de X (des 0 et des 1)
print("\nExemple de ligne X (premiere ligne):")
print(X_train.iloc[0][X_train.iloc[0] == 1])  # affiche seulement les symptomes presents (=1)

print("\nMaladie correspondante (y):", y_train.iloc[0])

# Encoder y : transformer "Diabetes", "Malaria", etc en 0, 1, 2...
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("\nExemple d'encodage :")
print("Maladie originale:", y_train.iloc[0], "-> code:", y_train_encoded[0])

# Sauvegarder l'encodeur pour plus tard (tres important, on en aura besoin
# pour retransformer les predictions en noms de maladies lisibles)
import joblib
joblib.dump(le, "label_encoder.pkl")
print("\nLabelEncoder sauvegarde dans label_encoder.pkl")