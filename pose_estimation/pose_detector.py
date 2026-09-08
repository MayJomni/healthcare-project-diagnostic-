"""
Détecteur de pose utilisant MediaPipe (compatible v0.x et v1.0+).
"""
import math
import logging

try:
    import numpy as np
except ImportError:
    logging.warning("NumPy n'est pas installé.")
    np = None

try:
    import cv2
except ImportError:
    logging.warning("OpenCV n'est pas installé. Veuillez l'installer avec 'pip install opencv-python'.")
    cv2 = None

# Détection de la version MediaPipe
mp = None
_MP_LEGACY = False
_MP_TASKS = False

try:
    import mediapipe as mp
    # MediaPipe 1.0+ utilise mp.tasks.vision
    if hasattr(mp, 'tasks'):
        _MP_TASKS = True
    # MediaPipe 0.x utilise mp.solutions.pose
    if hasattr(mp, 'solutions'):
        _MP_LEGACY = True
except ImportError:
    logging.warning("MediaPipe n'est pas installé. Veuillez l'installer avec 'pip install mediapipe'.")


LANDMARK_NAMES = {
    0: "nez",
    1: "oeil_interieur_gauche",
    2: "oeil_gauche",
    3: "oeil_exterieur_gauche",
    4: "oeil_interieur_droit",
    5: "oeil_droit",
    6: "oeil_exterieur_droit",
    7: "oreille_gauche",
    8: "oreille_droite",
    9: "bouche_gauche",
    10: "bouche_droite",
    11: "epaule_gauche",
    12: "epaule_droite",
    13: "coude_gauche",
    14: "coude_droit",
    15: "poignet_gauche",
    16: "poignet_droit",
    17: "auriculaire_gauche",
    18: "auriculaire_droit",
    19: "index_gauche",
    20: "index_droit",
    21: "pouce_gauche",
    22: "pouce_droit",
    23: "hanche_gauche",
    24: "hanche_droite",
    25: "genou_gauche",
    26: "genou_droit",
    27: "cheville_gauche",
    28: "cheville_droite",
    29: "talon_gauche",
    30: "talon_droit",
    31: "index_pied_gauche",
    32: "index_pied_droit"
}

# Connexions du squelette pour dessiner les lignes
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
]


class PoseDetector:
    """
    Classe pour détecter les poses humaines dans une image ou une vidéo.
    Compatible avec MediaPipe 0.x (solutions) et 1.0+ (tasks).
    """
    def __init__(self, backend='mediapipe'):
        """
        Initialise le détecteur de pose.

        Args:
            backend (str): Le backend à utiliser ('mediapipe').
        """
        self.backend = backend
        self.pose = None
        self._use_tasks = False

        if self.backend == 'mediapipe':
            if mp is None:
                raise ImportError("MediaPipe est requis pour le backend 'mediapipe'.")

            if _MP_TASKS and not _MP_LEGACY:
                # MediaPipe 1.0+ — nouvelle API tasks
                self._use_tasks = True
                self._init_tasks_api()
            elif _MP_LEGACY:
                # MediaPipe 0.x — ancienne API solutions
                self._use_tasks = False
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            else:
                raise ImportError("Version MediaPipe non reconnue.")
        else:
            raise ValueError("Backend non supporté. Choisissez 'mediapipe'.")

    def _init_tasks_api(self):
        """Initialise le PoseLandmarker avec l'API tasks de MediaPipe 1.0+."""
        import os
        import urllib.request

        # Télécharger le modèle si nécessaire
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "pose_landmarker_lite.task")

        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            logging.info(f"Téléchargement du modèle de pose : {url}")
            try:
                urllib.request.urlretrieve(url, model_path)
                logging.info("Modèle téléchargé avec succès.")
            except Exception as e:
                logging.error(f"Impossible de télécharger le modèle : {e}")
                raise

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self.pose = PoseLandmarker.create_from_options(options)

    def detect_pose(self, frame) -> dict:
        """
        Détecte la pose dans une image donnée.

        Args:
            frame (np.ndarray): L'image d'entrée (numpy array BGR).

        Returns:
            dict: Dictionnaire contenant {landmarks, nb_points_detectes, confiance_moyenne}.
        """
        resultat = {
            'landmarks': [],
            'nb_points_detectes': 0,
            'confiance_moyenne': 0.0
        }

        if cv2 is None or self.pose is None:
            return resultat

        if self._use_tasks:
            return self._detect_tasks(frame, resultat)
        else:
            return self._detect_legacy(frame, resultat)

    def _detect_legacy(self, frame, resultat):
        """Détection avec l'ancienne API mp.solutions.pose."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            total_vis = 0.0
            for i, lm in enumerate(results.pose_landmarks.landmark):
                nom = LANDMARK_NAMES.get(i, f"point_{i}")
                resultat['landmarks'].append({
                    'nom': nom,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibilite': lm.visibility
                })
                total_vis += lm.visibility
            resultat['nb_points_detectes'] = len(resultat['landmarks'])
            if resultat['nb_points_detectes'] > 0:
                resultat['confiance_moyenne'] = total_vis / resultat['nb_points_detectes']

        return resultat

    def _detect_tasks(self, frame, resultat):
        """Détection avec la nouvelle API mp.tasks.vision (MediaPipe 1.0+)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.pose.detect(mp_image)

        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            landmarks_list = detection_result.pose_landmarks[0]
            total_vis = 0.0

            for i, lm in enumerate(landmarks_list):
                nom = LANDMARK_NAMES.get(i, f"point_{i}")
                visibility = getattr(lm, 'visibility', 0.5)
                resultat['landmarks'].append({
                    'nom': nom,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibilite': visibility
                })
                total_vis += visibility

            resultat['nb_points_detectes'] = len(resultat['landmarks'])
            if resultat['nb_points_detectes'] > 0:
                resultat['confiance_moyenne'] = total_vis / resultat['nb_points_detectes']

        return resultat

    def calculate_angle(self, point_a: dict, point_b: dict, point_c: dict) -> float:
        """
        Calcule l'angle (en degrés) entre trois points de repère.
        Le point_b est le sommet de l'angle.

        Args:
            point_a (dict): Premier point contenant x et y.
            point_b (dict): Deuxième point (sommet) contenant x et y.
            point_c (dict): Troisième point contenant x et y.

        Returns:
            float: Angle en degrés (entre 0 et 180).
        """
        angle = math.degrees(
            math.atan2(point_c['y'] - point_b['y'], point_c['x'] - point_b['x']) -
            math.atan2(point_a['y'] - point_b['y'], point_a['x'] - point_b['x'])
        )
        angle = abs(angle)
        if angle > 180.0:
            angle = 360.0 - angle
        return angle

    def draw_pose(self, frame, landmarks: list):
        """
        Dessine le squelette sur l'image fournie.

        Args:
            frame (np.ndarray): L'image d'entrée.
            landmarks (list): Liste des points de repère détectés.

        Returns:
            np.ndarray: L'image avec le squelette superposé.
        """
        output_frame = frame.copy()
        if not landmarks or cv2 is None:
            return output_frame

        h, w, _ = frame.shape

        # Dessiner les connexions (lignes du squelette)
        for (a, b) in POSE_CONNECTIONS:
            if a < len(landmarks) and b < len(landmarks):
                pt1 = (int(landmarks[a]['x'] * w), int(landmarks[a]['y'] * h))
                pt2 = (int(landmarks[b]['x'] * w), int(landmarks[b]['y'] * h))
                cv2.line(output_frame, pt1, pt2, (0, 200, 200), 2)

        # Dessiner les points
        for lm in landmarks:
            cx, cy = int(lm['x'] * w), int(lm['y'] * h)
            cv2.circle(output_frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        return output_frame
