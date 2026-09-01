"""
Etape 7a : Validation croisee - vraie mesure de performance
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Charger les donnees
train = pd.read_csv("Training.csv")
X = train.drop(columns=["prognosis"])
y = train["prognosis"]

# Encoder y
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Les 3 modeles a comparer
modeles = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
}

print("=== VALIDATION CROISEE (5 folds) ===\n")
print("Cette fois, chaque modele est teste 5 fois sur des")
print("portions differentes des donnees, pas juste 1 seule fois.\n")

for nom, modele in modeles.items():
    scores = cross_val_score(modele, X, y_encoded, cv=5)
    print(f"{nom}:")
    print(f"  Scores sur les 5 folds : {[round(s, 3) for s in scores]}")
    print(f"  Moyenne : {round(scores.mean(), 4)}  (+/- {round(scores.std(), 4)})")
    print()