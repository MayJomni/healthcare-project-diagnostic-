"""
App Flask : prediction de maladie a partir de symptomes
Etapes 6-7 du plan : interface + ameliorations
 - 4 modeles compares (Decision Tree, Random Forest, Naive Bayes, XGBoost)
 - Explicabilite : symptomes qui ont le plus pese dans la decision
 - Historique des diagnostics sauvegarde en base SQLite
 - Feedback utilisateur (confiance + commentaire) pour etudier la relation
   entre consensus des modeles et confiance percue par l'utilisateur (HCI)
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import os
import sqlite3
import json
import re
from collections import Counter
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "historique.db")

# ---------------------------------------------------------
# Base de donnees : historique des diagnostics + feedback
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historique (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_heure TEXT NOT NULL,
            symptomes TEXT NOT NULL,
            resultats TEXT NOT NULL,
            consensus_maladie TEXT,
            consensus_nb_accord INTEGER,
            consensus_nb_total INTEGER
        )
    """)
    # Table de feedback : collecte quantitative (trust_score) + qualitative (commentaire)
    # liee a un diagnostic precis via historique_id (etude empirique HCI)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            historique_id INTEGER NOT NULL,
            date_heure TEXT NOT NULL,
            trust_score INTEGER NOT NULL,
            commentaire TEXT,
            FOREIGN KEY (historique_id) REFERENCES historique (id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# Chargement des donnees et entrainement des modeles
# ---------------------------------------------------------
train = pd.read_csv(os.path.join(BASE_DIR, "Training.csv"))

X_train = train.drop(columns=["prognosis"])
y_train = train["prognosis"]

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

SYMPTOMES = list(X_train.columns)


def bruiter_pour_augmentation(X_data, taux_effacement, nb_faux_symptomes, seed):
    """Cree une version bruitee du jeu d'entrainement (symptomes manquants/en trop)
    pour que le modele apprenne a reconnaitre des cas incomplets, comme dans la vraie vie."""
    rng = np.random.RandomState(seed)
    X_bruite = X_data.copy()
    for i in X_bruite.index:
        presents = X_bruite.columns[X_bruite.loc[i] == 1]
        n_a_effacer = int(len(presents) * taux_effacement)
        if n_a_effacer > 0:
            a_effacer = rng.choice(presents, n_a_effacer, replace=False)
            X_bruite.loc[i, a_effacer] = 0
        if nb_faux_symptomes > 0:
            absents = X_bruite.columns[X_bruite.loc[i] == 0]
            if len(absents) >= nb_faux_symptomes:
                faux = rng.choice(absents, nb_faux_symptomes, replace=False)
                X_bruite.loc[i, faux] = 1
    return X_bruite


# Augmentation de donnees : on ajoute des versions bruitees du dataset original
# pour que les modeles soient plus robustes face a des symptomes manquants/inexacts
# (cf. etape 7d : gain mesure de +19.8 points sur un test a 70% de bruit)
X_augmente = [X_train]
y_augmente = [y_train_encoded]
for idx, taux in enumerate([0.2, 0.4, 0.6]):
    X_bruite = bruiter_pour_augmentation(X_train, taux, nb_faux_symptomes=1, seed=42 + idx)
    X_augmente.append(X_bruite)
    y_augmente.append(y_train_encoded)

X_train_final = pd.concat(X_augmente, ignore_index=True)
y_train_final = np.concatenate(y_augmente)

modeles = {
    "Random Forest": RandomForestClassifier(random_state=42).fit(X_train_final, y_train_final),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, max_depth=12, min_samples_leaf=3
    ).fit(X_train_final, y_train_final),
    "Naive Bayes": GaussianNB().fit(X_train_final, y_train_final),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="mlogloss").fit(X_train_final, y_train_final),
}

# Modeles pour lesquels on peut extraire une importance de variable (explicabilite)
MODELES_AVEC_IMPORTANCE = ("Random Forest", "Decision Tree", "XGBoost")


def nom_lisible(symptome):
    return symptome.replace("_", " ").strip()


def get_explication(nom_modele, modele, symptomes_selectionnes):
    """Retourne les symptomes selectionnes qui ont le plus pese dans la decision,
    d'apres l'importance globale des variables du modele (feature_importances_)."""
    if nom_modele not in MODELES_AVEC_IMPORTANCE:
        return None

    importances = modele.feature_importances_
    poids = {SYMPTOMES[i]: importances[i] for i in range(len(SYMPTOMES))}

    # On ne garde que les symptomes que l'utilisateur a coches, tries par poids
    pertinents = [(s, poids[s]) for s in symptomes_selectionnes if s in poids]
    pertinents.sort(key=lambda x: x[1], reverse=True)

    top = pertinents[:3]
    return [{"symptome": nom_lisible(s), "poids": round(float(w), 4)} for s, w in top]


@app.route("/")
def index():
    symptomes_tries = sorted(SYMPTOMES, key=nom_lisible)
    return render_template("index.html", symptomes=symptomes_tries, nom_lisible=nom_lisible)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptomes_selectionnes = data.get("symptomes", [])

    if not symptomes_selectionnes:
        return jsonify({"error": "Selectionne au moins un symptome."}), 400

    vecteur = pd.DataFrame([[0] * len(SYMPTOMES)], columns=SYMPTOMES)
    for s in symptomes_selectionnes:
        if s in vecteur.columns:
            vecteur.at[0, s] = 1

    resultats = []
    for nom_modele, modele in modeles.items():
        if hasattr(modele, "predict_proba"):
            proba = modele.predict_proba(vecteur)[0]
            top3_idx = proba.argsort()[-3:][::-1]
            top3 = [
                {
                    "maladie": le.inverse_transform([idx])[0],
                    "confiance": round(float(proba[idx]) * 100, 1)
                }
                for idx in top3_idx
            ]
        else:
            pred_code = modele.predict(vecteur)[0]
            top3 = [{"maladie": le.inverse_transform([pred_code])[0], "confiance": None}]

        explication = get_explication(nom_modele, modele, symptomes_selectionnes)

        resultats.append({
            "modele": nom_modele,
            "top3": top3,
            "explication": explication,
        })

    # Calculer le consensus : combien de modeles sont d'accord sur le diagnostic n°1
    votes = [r["top3"][0]["maladie"] for r in resultats]
    maladie_majoritaire = max(set(votes), key=votes.count)
    nb_accord = votes.count(maladie_majoritaire)
    consensus = {
        "maladie": maladie_majoritaire,
        "nb_accord": nb_accord,
        "nb_total": len(votes),
    }

    # Sauvegarder dans l'historique (on garde l'id pour le lier au feedback)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """INSERT INTO historique
           (date_heure, symptomes, resultats, consensus_maladie, consensus_nb_accord, consensus_nb_total)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().strftime("%d/%m/%Y %H:%M"),
            json.dumps([nom_lisible(s) for s in symptomes_selectionnes], ensure_ascii=False),
            json.dumps(resultats, ensure_ascii=False),
            consensus["maladie"],
            consensus["nb_accord"],
            consensus["nb_total"],
        )
    )
    historique_id = cur.lastrowid
    conn.commit()
    conn.close()

    return jsonify({
        "historique_id": historique_id,
        "resultats": resultats,
        "nb_symptomes": len(symptomes_selectionnes),
        "consensus": consensus,
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    """Collecte le feedback utilisateur (confiance + commentaire) pour un diagnostic donne.
    Donnees quantitatives (trust_score 1-5) + qualitatives (commentaire libre)."""
    data = request.get_json()
    historique_id = data.get("historique_id")
    trust_score = data.get("trust_score")
    commentaire = (data.get("commentaire") or "").strip()

    if historique_id is None:
        return jsonify({"error": "historique_id manquant."}), 400

    try:
        trust_score = int(trust_score)
    except (TypeError, ValueError):
        return jsonify({"error": "trust_score doit etre un entier entre 1 et 5."}), 400

    if not (1 <= trust_score <= 5):
        return jsonify({"error": "trust_score doit etre entre 1 et 5."}), 400

    conn = sqlite3.connect(DB_PATH)
    existe = conn.execute("SELECT id FROM historique WHERE id = ?", (historique_id,)).fetchone()
    if existe is None:
        conn.close()
        return jsonify({"error": "historique_id introuvable."}), 404

    conn.execute(
        "INSERT INTO feedback (historique_id, date_heure, trust_score, commentaire) VALUES (?, ?, ?, ?)",
        (historique_id, datetime.now().strftime("%d/%m/%Y %H:%M"), trust_score, commentaire)
    )
    conn.commit()
    conn.close()

    return jsonify({"ok": True})


@app.route("/historique")
def historique():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM historique ORDER BY id DESC LIMIT 50").fetchall()
    conn.close()

    entries = []
    for row in rows:
        entries.append({
            "id": row["id"],
            "date_heure": row["date_heure"],
            "symptomes": json.loads(row["symptomes"]),
            "resultats": json.loads(row["resultats"]),
        })

    return render_template("historique.html", entries=entries)


@app.route("/historique/vider", methods=["POST"])
def vider_historique():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM historique")
    conn.execute("DELETE FROM feedback")
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# ---------------------------------------------------------
# Analyse quanti + quali du feedback (HCI / recherche empirique)
# ---------------------------------------------------------
STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "a", "au", "aux",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "ce", "cette",
    "ces", "mais", "ou", "donc", "car", "ne", "pas", "plus", "que", "qui",
    "est", "etait", "sont", "avec", "pour", "dans", "sur", "en", "se", "son",
    "sa", "ses", "mon", "ma", "mes", "ton", "ta", "tes",
}


def analyser_mots_cles(commentaires, top_n=10):
    """Analyse thematique tres simple : frequence des mots dans les commentaires libres."""
    compteur = Counter()
    for c in commentaires:
        mots = re.findall(r"[a-zàâäéèêëïîôöùûüç]{3,}", c.lower())
        for m in mots:
            if m not in STOPWORDS_FR:
                compteur[m] += 1
    return [{"mot": m, "occurrences": n} for m, n in compteur.most_common(top_n)]


@app.route("/analytics")
def analytics():
    """Retourne l'analyse quantitative (correlation consensus <-> confiance)
    et qualitative (mots-cles des commentaires) du feedback collecte."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT h.consensus_nb_accord, h.consensus_nb_total, f.trust_score, f.commentaire
        FROM feedback f
        JOIN historique h ON h.id = f.historique_id
    """).fetchall()
    conn.close()

    if not rows:
        return jsonify({
            "nb_reponses": 0,
            "message": "Aucun feedback collecte pour le moment.",
        })

    # Quantitatif : score de confiance moyen groupe par niveau de consensus
    par_consensus = {}
    trust_scores = []
    commentaires = []
    for r in rows:
        ratio = r["consensus_nb_accord"] / r["consensus_nb_total"] if r["consensus_nb_total"] else 0
        cle = f'{r["consensus_nb_accord"]}/{r["consensus_nb_total"]}'
        par_consensus.setdefault(cle, []).append(r["trust_score"])
        trust_scores.append(r["trust_score"])
        if r["commentaire"]:
            commentaires.append(r["commentaire"])

    moyenne_par_consensus = {
        cle: round(sum(scores) / len(scores), 2)
        for cle, scores in par_consensus.items()
    }

    # Correlation simple (Pearson) entre ratio de consensus et trust_score
    ratios = []
    scores = []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows2 = conn.execute("""
        SELECT h.consensus_nb_accord, h.consensus_nb_total, f.trust_score
        FROM feedback f JOIN historique h ON h.id = f.historique_id
    """).fetchall()
    conn.close()
    for r in rows2:
        if r["consensus_nb_total"]:
            ratios.append(r["consensus_nb_accord"] / r["consensus_nb_total"])
            scores.append(r["trust_score"])

    correlation = None
    if len(ratios) >= 2 and np.std(ratios) > 0 and np.std(scores) > 0:
        correlation = round(float(np.corrcoef(ratios, scores)[0, 1]), 3)

    return jsonify({
        "nb_reponses": len(rows),
        "trust_score_moyen": round(sum(trust_scores) / len(trust_scores), 2),
        "moyenne_par_niveau_consensus": moyenne_par_consensus,
        "correlation_consensus_confiance": correlation,
        "mots_cles_commentaires": analyser_mots_cles(commentaires),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)