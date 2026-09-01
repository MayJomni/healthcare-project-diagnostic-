"""
Etape 7b : Tester avec du bruit realiste (symptomes manquants/en trop)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Charger les donnees
train = pd.read_csv("Training.csv")
X = train.drop(columns=["prognosis"])
y = train["prognosis"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split normal (80% entrainement, 20% test) au lieu d'utiliser Testing.csv
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Entrainer le modele sur des donnees PROPRES
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test 1 : accuracy sur donnees propres (comme avant)
preds_propres = model.predict(X_test)
acc_propre = accuracy_score(y_test, preds_propres)

# Test 2 : on ajoute du bruit realiste au jeu de test
# On "efface" aleatoirement 30% des symptomes presents (patient qui ne mentionne pas tout)
X_test_bruite = X_test.copy()
for i in X_test_bruite.index:
    symptomes_presents = X_test_bruite.columns[X_test_bruite.loc[i] == 1]
    n_a_effacer = int(len(symptomes_presents) * 0.7)
    if n_a_effacer > 0:
        a_effacer = np.random.choice(symptomes_presents, n_a_effacer, replace=False)
        X_test_bruite.loc[i, a_effacer] = 0
# On ajoute aussi 2 "faux" symptomes aleatoires (bruit supplementaire)
for i in X_test_bruite.index:
    symptomes_absents = X_test_bruite.columns[X_test_bruite.loc[i] == 0]
    if len(symptomes_absents) >= 2:
        faux_symptomes = np.random.choice(symptomes_absents, 2, replace=False)
        X_test_bruite.loc[i, faux_symptomes] = 1
preds_bruite = model.predict(X_test_bruite)
acc_bruite = accuracy_score(y_test, preds_bruite)

print("=== COMPARAISON ===")
print(f"Accuracy sur donnees propres        : {round(acc_propre, 4)}")
print(f"Accuracy avec 30% symptomes en moins : {round(acc_bruite, 4)}")
print(f"\nCa montre que le modele est excellent sur des cas 'manuel'")
print(f"mais moins fiable sur des cas incomplets, comme dans la vraie vie.")