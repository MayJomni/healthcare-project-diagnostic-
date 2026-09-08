"""
App Flask : plateforme de diagnostic et d'analyse de sante
Integre 6 modules :
 - Diagnostic ML (4 modeles : Decision Tree, Random Forest, Naive Bayes, XGBoost)
 - Diagnostic LLM (OpenAI / Anthropic / mode simulation)
 - Analyse NLP (clustering semantique, modelisation thematique, mots-cles)
 - Analyse d'ecosystemes (quadruple helice, classification, scraping)
 - Estimation de pose (detection posturale, analyse de sante, video temps reel)
 - Dashboard integre avec navigation unifiee
"""
from flask import Flask, render_template, request, jsonify, Response
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

# Configuration centralisee
import config

app = Flask(__name__)

BASE_DIR = config.BASE_DIR
DB_PATH = config.DB_PATH

# ---------------------------------------------------------
# Imports conditionnels des nouveaux modules
# ---------------------------------------------------------

# Module LLM
llm_engine = None
if config.ENABLE_LLM:
    try:
        from llm_diagnostic import LLMDiagnosticEngine
        llm_engine = LLMDiagnosticEngine()
        print("* Module LLM charge avec succes.")
    except Exception as e:
        print(f"! Module LLM non disponible : {e}")

# Module NLP
nlp_clusterer = None
nlp_topic_modeler = None
nlp_keyword_mapper = None
if config.ENABLE_NLP:
    try:
        from nlp_analysis.semantic_clustering import SemanticClusterer
        from nlp_analysis.topic_modeling import TopicModeler
        from nlp_analysis.keyword_mapping import KeywordMapper
        nlp_clusterer = SemanticClusterer()
        nlp_topic_modeler = TopicModeler()
        nlp_keyword_mapper = KeywordMapper()
        print("* Module NLP charge avec succes.")
    except Exception as e:
        print(f"! Module NLP non disponible : {e}")

# Module Ecosystemes
eco_classifier = None
eco_helix = None
eco_scraper = None
if config.ENABLE_ECOSYSTEM:
    try:
        from ecosystem_analysis.ecosystem_classifier import EcosystemClassifier
        from ecosystem_analysis.quadruple_helix import QuadrupleHelixAnalyzer
        from ecosystem_analysis.report_scraper import ReportScraper
        eco_classifier = EcosystemClassifier()
        eco_helix = QuadrupleHelixAnalyzer()
        eco_scraper = ReportScraper()
        print("* Module Ecosystemes charge avec succes.")
    except Exception as e:
        print(f"! Module Ecosystemes non disponible : {e}")

# Module Pose
pose_detector = None
health_analyzer = None
video_feed_generator = None
if config.ENABLE_POSE:
    try:
        from pose_estimation.pose_detector import PoseDetector
        from pose_estimation.health_analyzer import HealthAnalyzer
        from pose_estimation.video_stream import VideoStream, generate_video_feed
        pose_detector = PoseDetector()
        health_analyzer = HealthAnalyzer()
        print("* Module Estimation de pose charge avec succes.")
    except Exception as e:
        print(f"! Module Estimation de pose non disponible : {e}")


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
    # Migration : ajouter les colonnes consensus si elles n'existent pas (ancien schema)
    colonnes_existantes = [row[1] for row in conn.execute("PRAGMA table_info(historique)").fetchall()]
    for col, col_type in [("consensus_maladie", "TEXT"), ("consensus_nb_accord", "INTEGER"), ("consensus_nb_total", "INTEGER")]:
        if col not in colonnes_existantes:
            conn.execute(f"ALTER TABLE historique ADD COLUMN {col} {col_type}")
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
# Chargement des donnees et entrainement des modeles ML
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

    pertinents = [(s, poids[s]) for s in symptomes_selectionnes if s in poids]
    pertinents.sort(key=lambda x: x[1], reverse=True)

    top = pertinents[:3]
    return [{"symptome": nom_lisible(s), "poids": round(float(w), 4)} for s, w in top]


# =========================================================
# ROUTES — Pages principales
# =========================================================

@app.route("/")
def index():
    symptomes_tries = sorted(SYMPTOMES, key=nom_lisible)
    return render_template("index.html", symptomes=symptomes_tries, nom_lisible=nom_lisible)


@app.route("/dashboard")
def dashboard():
    """Tableau de bord principal avec acces a tous les modules."""
    return render_template("dashboard.html")


@app.route("/nlp")
def nlp_page():
    """Page d'analyse NLP."""
    return render_template("nlp_dashboard.html")


@app.route("/pose")
def pose_page():
    """Page d'estimation de pose."""
    return render_template("pose_view.html")


@app.route("/ecosystem")
def ecosystem_page():
    """Page d'analyse d'ecosystemes."""
    return render_template("ecosystem_report.html")


# =========================================================
# ROUTES — Diagnostic ML (existant)
# =========================================================

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

    # Calculer le consensus
    votes = [r["top3"][0]["maladie"] for r in resultats]
    maladie_majoritaire = max(set(votes), key=votes.count)
    nb_accord = votes.count(maladie_majoritaire)
    consensus = {
        "maladie": maladie_majoritaire,
        "nb_accord": nb_accord,
        "nb_total": len(votes),
    }

    # Sauvegarder dans l'historique
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


# =========================================================
# ROUTES — Diagnostic LLM (Module 1)
# =========================================================

@app.route("/predict_llm", methods=["POST"])
def predict_llm():
    """Diagnostic via LLM (OpenAI/Anthropic) ou mode simulation."""
    if not llm_engine:
        return jsonify({"error": "Le module LLM n'est pas active."}), 503

    data = request.get_json()
    symptomes_selectionnes = data.get("symptomes", [])

    if not symptomes_selectionnes:
        return jsonify({"error": "Selectionne au moins un symptome."}), 400

    symptomes_lisibles = [nom_lisible(s) for s in symptomes_selectionnes]
    resultat = llm_engine.diagnostiquer(symptomes_lisibles)

    return jsonify({
        "modele": "LLM",
        "mode": "simulation" if llm_engine.mode_simulation else config.LLM_PROVIDER,
        "resultat": resultat,
    })


# =========================================================
# ROUTES — Feedback (existant)
# =========================================================

@app.route("/feedback", methods=["POST"])
def feedback():
    """Collecte le feedback utilisateur (confiance + commentaire) pour un diagnostic donne."""
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


# =========================================================
# ROUTES — Historique (existant)
# =========================================================

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


# =========================================================
# ROUTES — Analyse NLP (Module 2)
# =========================================================

@app.route("/api/nlp/cluster", methods=["POST"])
def api_nlp_cluster():
    """Clustering semantique de textes."""
    if not nlp_clusterer:
        return jsonify({"error": "Le module NLP n'est pas active."}), 503

    data = request.get_json()
    textes = data.get("textes", [])
    n_clusters = data.get("n_clusters", 5)

    if not textes or len(textes) < 2:
        return jsonify({"error": "Fournissez au moins 2 textes."}), 400

    try:
        resultats = nlp_clusterer.cluster(textes, n_clusters=min(n_clusters, len(textes)))
        return jsonify(resultats)
    except Exception as e:
        return jsonify({"error": f"Erreur lors du clustering : {str(e)}"}), 500


@app.route("/api/nlp/topics", methods=["POST"])
def api_nlp_topics():
    """Modelisation thematique de textes."""
    if not nlp_topic_modeler:
        return jsonify({"error": "Le module NLP n'est pas active."}), 503

    data = request.get_json()
    textes = data.get("textes", [])
    n_topics = data.get("n_topics", 5)

    if not textes or len(textes) < 2:
        return jsonify({"error": "Fournissez au moins 2 textes."}), 400

    try:
        resultats = nlp_topic_modeler.extract_topics(textes, n_topics=n_topics)
        return jsonify({"topics": resultats})
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la modelisation thematique : {str(e)}"}), 500


@app.route("/api/nlp/keywords", methods=["POST"])
def api_nlp_keywords():
    """Extraction de mots-cles."""
    if not nlp_keyword_mapper:
        return jsonify({"error": "Le module NLP n'est pas active."}), 503

    data = request.get_json()
    textes = data.get("textes", [])
    top_n = data.get("top_n", 20)

    if not textes:
        return jsonify({"error": "Fournissez au moins un texte."}), 400

    try:
        keywords_tfidf = nlp_keyword_mapper.extract_keywords_tfidf(textes, top_n=top_n)
        cooccurrence = nlp_keyword_mapper.build_cooccurrence_map(textes, top_n=top_n)
        return jsonify({
            "mots_cles_tfidf": keywords_tfidf,
            "cooccurrence": cooccurrence,
        })
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'extraction de mots-cles : {str(e)}"}), 500


# =========================================================
# ROUTES — Analyse d'ecosystemes (Module 3)
# =========================================================

@app.route("/api/ecosystem/analyze", methods=["POST"])
def api_ecosystem_analyze():
    """Analyse d'ecosysteme complete (classification + quadruple helice)."""
    if not eco_classifier or not eco_helix:
        return jsonify({"error": "Le module Ecosystemes n'est pas active."}), 503

    data = request.get_json()
    texte = data.get("texte", "")
    textes = data.get("textes", [])

    if texte:
        textes = [texte]

    if not textes:
        return jsonify({"error": "Fournissez au moins un texte a analyser."}), 400

    try:
        classification = eco_classifier.analyze_corpus(textes)
        rapport_helix = eco_helix.generate_report(textes)

        return jsonify({
            "classification_ecosysteme": classification,
            "analyse_quadruple_helice": rapport_helix,
        })
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'analyse : {str(e)}"}), 500


@app.route("/api/ecosystem/scrape", methods=["POST"])
def api_ecosystem_scrape():
    """Scraping de rapports et articles pour l'analyse d'ecosystemes."""
    if not eco_scraper:
        return jsonify({"error": "Le module Ecosystemes n'est pas active."}), 503

    data = request.get_json()
    query = data.get("query", "")
    source = data.get("source", "techcrunch")
    max_articles = data.get("max_articles", 5)

    if not query:
        return jsonify({"error": "Fournissez une requete de recherche."}), 400

    try:
        articles = eco_scraper.scrape_tech_articles(
            query=query, source=source, max_articles=max_articles
        )
        return jsonify({"articles": articles})
    except Exception as e:
        return jsonify({"error": f"Erreur lors du scraping : {str(e)}"}), 500


# =========================================================
# ROUTES — Estimation de pose (Module 4)
# =========================================================

@app.route("/video_feed")
def video_feed():
    """Flux video avec overlay d'estimation de pose."""
    if not pose_detector or not health_analyzer:
        return jsonify({"error": "Le module d'estimation de pose n'est pas active."}), 503

    return Response(
        generate_video_feed(pose_detector, health_analyzer),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/pose/status")
def api_pose_status():
    """Statut de l'estimation de pose."""
    if not pose_detector or not health_analyzer:
        return jsonify({"error": "Le module d'estimation de pose n'est pas active."}), 503

    return jsonify({
        "module_actif": True,
        "backend": "mediapipe",
        "message": "Le module de pose est pret. Utilisez /video_feed pour le flux video.",
    })


# =========================================================
# ROUTES — Analytics (existant)
# =========================================================

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
        mots = re.findall(r"[a-z\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb\u00ef\u00ee\u00f4\u00f6\u00f9\u00fb\u00fc\u00e7]{3,}", c.lower())
        for m in mots:
            if m not in STOPWORDS_FR:
                compteur[m] += 1
    return [{"mot": m, "occurrences": n} for m, n in compteur.most_common(top_n)]


@app.route("/analytics")
def analytics():
    """Retourne l'analyse quantitative et qualitative du feedback collecte."""
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

    par_consensus = {}
    trust_scores = []
    commentaires = []
    for r in rows:
        cle = f'{r["consensus_nb_accord"]}/{r["consensus_nb_total"]}'
        par_consensus.setdefault(cle, []).append(r["trust_score"])
        trust_scores.append(r["trust_score"])
        if r["commentaire"]:
            commentaires.append(r["commentaire"])

    moyenne_par_consensus = {
        cle: round(sum(scores) / len(scores), 2)
        for cle, scores in par_consensus.items()
    }

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


# =========================================================
# Route utilitaire : statut de tous les modules
# =========================================================

@app.route("/api/status")
def api_status():
    """Retourne le statut de tous les modules integres."""
    return jsonify({
        "modules": {
            "diagnostic_ml": True,
            "diagnostic_llm": llm_engine is not None,
            "llm_mode": ("simulation" if llm_engine and llm_engine.mode_simulation else config.LLM_PROVIDER) if llm_engine else "desactive",
            "nlp": nlp_clusterer is not None,
            "ecosystemes": eco_classifier is not None,
            "estimation_pose": pose_detector is not None,
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)