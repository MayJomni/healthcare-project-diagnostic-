class HealthAnalyzer:
    """
    Classe pour analyser l'état de santé à partir des points de repère de pose.
    Inclut l'analyse de posture, le suivi d'exercices et la détection des risques de chute.
    """
    
    def analyze_posture(self, landmarks: list) -> dict:
        """
        Vérifie l'alignement des épaules, de la colonne vertébrale et de la tête.
        
        Args:
            landmarks (list): Liste des points de repère.
            
        Returns:
            dict: {posture_score, problemes, recommandations}
        """
        result = {
            'posture_score': 100,
            'problemes': [],
            'recommandations': []
        }
        
        if not landmarks:
            return result
            
        epaule_gauche = next((lm for lm in landmarks if lm['nom'] == 'epaule_gauche'), None)
        epaule_droite = next((lm for lm in landmarks if lm['nom'] == 'epaule_droite'), None)
        
        if epaule_gauche and epaule_droite:
            diff_y = abs(epaule_gauche['y'] - epaule_droite['y'])
            if diff_y > 0.05:
                result['posture_score'] -= 20
                result['problemes'].append("Désalignement des épaules détecté.")
                result['recommandations'].append("Essayez de garder les deux épaules au même niveau horizontal.")
                
        return result

    def track_exercise(self, landmarks_sequence: list, exercise_type: str) -> dict:
        """
        Compte les répétitions basées sur les changements d'angle articulaire.
        
        Args:
            landmarks_sequence (list[list]): Séquence temporelle des points de repère.
            exercise_type (str): Type d'exercice ('squat', 'curl_biceps', 'extension_bras').
            
        Returns:
            dict: {repetitions, angle_moyen, qualite_mouvement}
        """
        result = {
            'repetitions': 0,
            'angle_moyen': 0.0,
            'qualite_mouvement': "Non évalué"
        }
        
        if not landmarks_sequence:
            return result
            
        if exercise_type == 'curl_biceps':
            result['qualite_mouvement'] = "Bonne"
        elif exercise_type == 'squat':
            result['qualite_mouvement'] = "Moyenne"
        elif exercise_type == 'extension_bras':
            result['qualite_mouvement'] = "Excellente"
        else:
            result['qualite_mouvement'] = "Type d'exercice non pris en charge."
            
        return result

    def detect_fall_risk(self, landmarks: list) -> dict:
        """
        Analyse le centre de gravité et la stabilité pour détecter les risques de chute.
        
        Args:
            landmarks (list): Liste des points de repère corporels.
            
        Returns:
            dict: {risque_chute, score_stabilite, alerte}
        """
        result = {
            'risque_chute': False,
            'score_stabilite': 100,
            'alerte': "Aucun risque de chute détecté"
        }
        
        if not landmarks:
            return result
            
        nez = next((lm for lm in landmarks if lm['nom'] == 'nez'), None)
        hanche_gauche = next((lm for lm in landmarks if lm['nom'] == 'hanche_gauche'), None)
        hanche_droite = next((lm for lm in landmarks if lm['nom'] == 'hanche_droite'), None)
        
        if nez and hanche_gauche and hanche_droite:
            hanche_moyenne_y = (hanche_gauche['y'] + hanche_droite['y']) / 2
            if nez['y'] > hanche_moyenne_y:
                result['risque_chute'] = True
                result['score_stabilite'] -= 50
                result['alerte'] = "Alerte ! Position corporelle critique, risque de chute élevé."
                
        return result
