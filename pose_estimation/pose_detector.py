import math
import logging

try:
    import numpy as np
except ImportError:
    logging.warning("NumPy n'est pas installé.")
    np = None

try:
    import mediapipe as mp
except ImportError:
    logging.warning("MediaPipe n'est pas installé. Veuillez l'installer avec 'pip install mediapipe'.")
    mp = None

try:
    import tensorflow as tf
except ImportError:
    logging.warning("TensorFlow n'est pas installé. La détection MoveNet ne sera pas disponible.")
    tf = None

try:
    import cv2
except ImportError:
    logging.warning("OpenCV n'est pas installé. Veuillez l'installer avec 'pip install opencv-python'.")
    cv2 = None

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

class PoseDetector:
    """
    Classe pour détecter les poses humaines dans une image ou une vidéo.
    Supporte MediaPipe (par défaut) ou TensorFlow MoveNet.
    """
    def __init__(self, backend='mediapipe'):
        """
        Initialise le détecteur de pose.
        
        Args:
            backend (str): Le backend à utiliser ('mediapipe' ou 'tensorflow').
        """
        self.backend = backend
        if self.backend == 'mediapipe':
            if mp is None:
                raise ImportError("MediaPipe est requis pour le backend 'mediapipe'.")
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.mp_drawing = mp.solutions.drawing_utils
        elif self.backend == 'tensorflow':
            if tf is None:
                raise ImportError("TensorFlow est requis pour le backend 'tensorflow'.")
            # Initialisation pour TensorFlow MoveNet
            pass
        else:
            raise ValueError("Backend non supporté. Choisissez 'mediapipe' ou 'tensorflow'.")

    def detect_pose(self, frame) -> dict:
        """
        Détecte la pose dans une image donnée.
        
        Args:
            frame (np.ndarray): L'image d'entrée (numpy array).
            
        Returns:
            dict: Dictionnaire contenant {landmarks, nb_points_detectes, confiance_moyenne}.
        """
        resultat = {
            'landmarks': [],
            'nb_points_detectes': 0,
            'confiance_moyenne': 0.0
        }

        if self.backend == 'mediapipe' and cv2 is not None:
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
        if self.backend == 'mediapipe' and landmarks and cv2 is not None:
            h, w, _ = frame.shape
            for lm in landmarks:
                cx, cy = int(lm['x'] * w), int(lm['y'] * h)
                cv2.circle(output_frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return output_frame
