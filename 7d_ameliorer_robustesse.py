"""
Etape 7d : Ameliorer la robustesse par augmentation de donnees
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

train = pd.read_csv("Training.csv")
X = train.drop(columns=["prognosis"])
y = train["prognosis"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


def bruiter(X_data, taux_effacement, nb_faux_symptomes=2):
    """Simule des rapports de symptomes incomplets/inexacts."""
    X_bruite = X_data.copy()
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


# --- MODELE 1 : entraine normalement (comme avant) ---
model_normal = RandomForestClassifier(random_state=42)
model_normal.fit(X_train, y_train)

# --- MODELE 2 : entraine avec AUGMENTATION DE DONNEES ---
# On cree plusieurs versions bruitees de chaque ligne d'entrainement,
# a differents niveaux de bruit, pour que le modele apprenne des cas incomplets.
X_augmente = [X_train]
y_augmente = [y_train]

for taux in [0.2, 0.4, 0.6]:
    X_bruite_train = bruiter(X_train, taux, nb_faux_symptomes=1)
    X_augmente.append(X_bruite_train)
    y_augmente.append(y_train)

X_train_augmente = pd.concat(X_augmente, ignore_index=True)
y_train_augmente = np.concatenate(y_augmente)

print(f"Taille du dataset original : {len(X_train)} lignes")
print(f"Taille du dataset augmente  : {len(X_train_augmente)} lignes\n")

model_robuste = RandomForestClassifier(random_state=42)
model_robuste.fit(X_train_augmente, y_train_augmente)

# --- COMPARAISON sur un jeu de test bruite a 70% (le pire cas qu'on avait) ---
X_test_bruite = bruiter(X_test, 0.7, nb_faux_symptomes=2)

acc_normal = accuracy_score(y_test, model_normal.predict(X_test_bruite))
acc_robuste = accuracy_score(y_test, model_robuste.predict(X_test_bruite))

print("=== COMPARAISON sur test bruite a 70% ===")
print(f"Modele normal (entraine sur donnees propres)   : {round(acc_normal, 4)}")
print(f"Modele robuste (entraine avec augmentation)     : {round(acc_robuste, 4)}")
print(f"\nGain : {round((acc_robuste - acc_normal) * 100, 1)} points de pourcentage")