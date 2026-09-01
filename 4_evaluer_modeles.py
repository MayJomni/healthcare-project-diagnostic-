"""
Etape 4 : Evaluer et comparer les modeles
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Charger les donnees
train = pd.read_csv("Training.csv")
test = pd.read_csv("Testing.csv")

X_train = train.drop(columns=["prognosis"])
y_train = train["prognosis"]
X_test = test.drop(columns=["prognosis"])
y_test = test["prognosis"]

# Encoder y
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Entrainer les 3 modeles
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train_encoded)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train_encoded)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train_encoded)

# Predire sur Testing.csv
dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
nb_preds = nb_model.predict(X_test)

# Comparer les accuracy (% de bonnes predictions)
print("=== ACCURACY (precision globale) ===")
print("Decision Tree :", accuracy_score(y_test_encoded, dt_preds))
print("Random Forest :", accuracy_score(y_test_encoded, rf_preds))
print("Naive Bayes   :", accuracy_score(y_test_encoded, nb_preds))

# Rapport detaille pour le meilleur modele (Random Forest en general)
print("\n=== RAPPORT DETAILLE - Random Forest ===")
print(classification_report(y_test_encoded, rf_preds, target_names=le.classes_, zero_division=0))