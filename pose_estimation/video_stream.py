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


class VideoStream:
    """
    Gère la capture de flux vidéo (webcam ou autre source).
    """
    def __init__(self, source=0):
        """
        Initialise le flux vidéo avec la source spécifiée.
        
        Args:
            source (int|str): Index de la webcam (par défaut 0) ou chemin de fichier vidéo.
        """
        self.source = source
        self.cap = None

    def start_stream(self):
        """
        Démarre la capture des images depuis la source.
        """
        if cv2 is None:
            raise ImportError("Impossible de démarrer le flux : OpenCV est requis.")
        self.cap = cv2.VideoCapture(self.source)

    def get_frame(self) -> tuple:
        """
        Récupère l'image actuelle.
        
        Returns:
            tuple[bool, np.ndarray]: (Succès de lecture, l'image lue en format numpy array).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def stop_stream(self):
        """
        Arrête la capture et libère la caméra.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def generate_video_feed(pose_detector, health_analyzer):
    """
    Fonction génératrice Flask pour fournir des images JPEG avec superposition
    du squelette et des informations de posture.
    
    Args:
        pose_detector: Une instance de PoseDetector.
        health_analyzer: Une instance de HealthAnalyzer.
        
    Yields:
        Des trames au format multipart MIME pour le streaming vidéo (Response Flask).
    """
    if cv2 is None:
        yield b"Erreur: OpenCV est manquant."
        return

    stream = VideoStream()
    stream.start_stream()
    
    try:
        while True:
            ret, frame = stream.get_frame()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
                
            pose_result = pose_detector.detect_pose(frame)
            landmarks = pose_result.get('landmarks', [])
            
            frame = pose_detector.draw_pose(frame, landmarks)
            
            if landmarks:
                posture_data = health_analyzer.analyze_posture(landmarks)
                fall_data = health_analyzer.detect_fall_risk(landmarks)
                
                score = posture_data.get('posture_score', 100)
                cv2.putText(frame, f"Score Posture: {score}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if fall_data.get('risque_chute', False):
                    alerte = fall_data.get('alerte', 'Alerte chute')
                    cv2.putText(frame, alerte, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    finally:
        stream.stop_stream()
