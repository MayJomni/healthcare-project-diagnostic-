"""
Etape 7c : Courbe de robustesse - accuracy selon le niveau de bruit
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

# Charger les donnees
train = pd.read_csv("Training.csv")
X = train.drop(columns=["prognosis"])
y = train["prognosis"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

def bruiter(X_test, taux_effacement, nb_faux_symptomes):
    """Efface un pourcentage de vrais symptomes et ajoute des faux."""
    X_bruite = X_test.copy()
    for i in X_bruite.index:
        presents = X_bruite.columns[X_bruite.loc[i] == 1]
        n_a_effacer = int(len(presents) * taux_effacement)
        if n_a_effacer > 0:
            a_effacer = np.random.choice(presents, n_a_effacer, replace=False)
            X_bruite.loc[i, a_effacer] = 0

        if nb_faux_symptomes > 0:
            absents = X_bruite.columns[X_bruite.loc[i] == 0]
            if len(absents) >= nb_faux_symptomes:
                faux = np.random.choice(absents, nb_faux_symptomes, replace=False)
                X_bruite.loc[i, faux] = 1
    return X_bruite

# Tester plusieurs niveaux de bruit
niveaux = [0.0, 0.1, 0.3, 0.5, 0.7]
accuracies = []

print("=== COURBE DE ROBUSTESSE ===\n")
for niveau in niveaux:
    X_bruite = bruiter(X_test, niveau, nb_faux_symptomes=2 if niveau > 0 else 0)
    preds = model.predict(X_bruite)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)
    print(f"Bruit {int(niveau*100)}% -> Accuracy : {round(acc, 4)}")

# Generer le graphique
plt.figure(figsize=(8, 5))
plt.plot([int(n*100) for n in niveaux], accuracies, marker='o', linewidth=2, color='#B5472F')
plt.title("Robustesse du modele face au bruit dans les symptomes")
plt.xlabel("% de symptomes effaces (+ 2 faux symptomes ajoutes)")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("courbe_robustesse.png", dpi=150)
print("\nGraphique sauvegarde : courbe_robustesse.png")
plt.show()