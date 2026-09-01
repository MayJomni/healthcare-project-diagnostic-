"""
Etape 3 : Entrainer plusieurs modeles
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Charger les donnees
train = pd.read_csv("Training.csv")
test = pd.read_csv("Testing.csv")

X_train = train.drop(columns=["prognosis"])
y_train = train["prognosis"]
X_test = test.drop(columns=["prognosis"])
y_test = test["prognosis"]

# Encoder y (comme a l'etape 2)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Modele 1 : Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train_encoded)
print("Decision Tree entraine !")

# Modele 2 : Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train_encoded)
print("Random Forest entraine !")

# Modele 3 : Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train_encoded)
print("Naive Bayes entraine !")

print("\nLes 3 modeles sont prets pour l'etape 4 (evaluation) !")