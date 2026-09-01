"""
Etape 5 : Sauvegarder le meilleur modele
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

train = pd.read_csv("Training.csv")
X_train = train.drop(columns=["prognosis"])
y_train = train["prognosis"]

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train_encoded)

# Sauvegarder le modele et l'encodeur
joblib.dump(rf_model, "modele_random_forest.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(list(X_train.columns), "liste_symptomes.pkl")  # utile pour l'interface plus tard

print("Modele sauvegarde : modele_random_forest.pkl")
print("Encodeur sauvegarde : label_encoder.pkl")
print("Liste des symptomes sauvegardee : liste_symptomes.pkl")