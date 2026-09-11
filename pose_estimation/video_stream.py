"""
Flux vidéo avec estimation de pose en temps réel.
Les statistiques sont stockées dans latest_pose_stats pour être lues par l'API.
"""
import time
import logging

try:
    import cv2
except ImportError:
    logging.warning("OpenCV n'est pas installé. La classe VideoStream nécessite OpenCV.")
    cv2 = None

try:
    import numpy as np
except ImportError:
    logging.warning("NumPy n'est pas installé.")
    np = None

# Statistiques temps réel partagées avec l'API Flask
latest_pose_stats = {
    "posture_score": None,
    "problemes": [],
    "recommandations": [],
    "risque_chute": False,
    "alerte": "",
    "nb_points": 0,
    "confiance": 0.0,
    "actif": False,
}


class VideoStream:
    """Gère la capture de flux vidéo (webcam ou autre source)."""
    def __init__(self, source=0):
        self.source = source
        self.cap = None

    def start_stream(self):
        if cv2 is None:
            raise ImportError("Impossible de démarrer le flux : OpenCV est requis.")
        self.cap = cv2.VideoCapture(self.source)

    def get_frame(self) -> tuple:
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def stop_stream(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def generate_video_feed(pose_detector, health_analyzer):
    """
    Génératrice Flask : images JPEG avec squelette + stats temps réel.
    Met à jour latest_pose_stats à chaque frame analysée.
    """
    global latest_pose_stats

    if cv2 is None:
        yield b"Erreur: OpenCV est manquant."
        return

    stream = VideoStream()
    stream.start_stream()
    latest_pose_stats["actif"] = True

    try:
        while True:
            ret, frame = stream.get_frame()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            pose_result = pose_detector.detect_pose(frame)
            landmarks = pose_result.get('landmarks', [])

            frame = pose_detector.draw_pose(frame, landmarks)

            # Mettre à jour les stats temps réel
            latest_pose_stats["nb_points"] = pose_result.get('nb_points_detectes', 0)
            latest_pose_stats["confiance"] = round(pose_result.get('confiance_moyenne', 0.0), 2)

            if landmarks:
                posture_data = health_analyzer.analyze_posture(landmarks)
                fall_data = health_analyzer.detect_fall_risk(landmarks)

                score = posture_data.get('posture_score', 100)
                latest_pose_stats["posture_score"] = score
                latest_pose_stats["problemes"] = posture_data.get('problemes', [])
                latest_pose_stats["recommandations"] = posture_data.get('recommandations', [])
                latest_pose_stats["risque_chute"] = fall_data.get('risque_chute', False)
                latest_pose_stats["alerte"] = fall_data.get('alerte', '')

                # Afficher le score sur la vidéo
                color = (0, 255, 0) if score >= 70 else (0, 165, 255) if score >= 40 else (0, 0, 255)
                cv2.putText(frame, f"Posture: {score}/100", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if latest_pose_stats["risque_chute"]:
                    cv2.putText(frame, "ALERTE CHUTE!", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Afficher les problèmes détectés
                for i, prob in enumerate(latest_pose_stats["problemes"][:3]):
                    cv2.putText(frame, prob, (10, 95 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            else:
                latest_pose_stats["posture_score"] = None

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        latest_pose_stats["actif"] = False
        stream.stop_stream()
