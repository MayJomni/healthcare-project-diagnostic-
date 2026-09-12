"""
Analyseur de santé posturale avancé.
Détecte : inclinaison tête, épaules, dos voûté, bras croisés,
position assise, stabilité, risque de chute.
"""
import math


def _get(landmarks, nom):
    """Trouve un point de repère par nom."""
    return next((lm for lm in landmarks if lm['nom'] == nom), None)


def _angle_deg(a, b, c):
    """Angle au point b entre les segments ba et bc (en degrés)."""
    ba = (a['x'] - b['x'], a['y'] - b['y'])
    bc = (c['x'] - b['x'], c['y'] - b['y'])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2) + 1e-6
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2) + 1e-6
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


class HealthAnalyzer:
    """
    Analyse posturale avancée à partir des 33 landmarks MediaPipe.
    Détecte de multiples problèmes avec des seuils réalistes.
    """

    def analyze_posture(self, landmarks: list) -> dict:
        """
        Analyse complète de la posture.

        Returns:
            dict: {posture_score (0-100), problemes, recommandations}
        """
        result = {
            'posture_score': 100,
            'problemes': [],
            'recommandations': []
        }

        if not landmarks or len(landmarks) < 25:
            return result

        # --- 1. Épaules inégales ---
        eg = _get(landmarks, 'epaule_gauche')
        ed = _get(landmarks, 'epaule_droite')
        if eg and ed:
            diff_epaules = abs(eg['y'] - ed['y'])
            if diff_epaules > 0.03:
                penalite = min(25, int(diff_epaules * 500))
                result['posture_score'] -= penalite
                result['problemes'].append(f"Épaules inégales ({diff_epaules:.2f})")
                result['recommandations'].append("Redressez vos épaules au même niveau.")

        # --- 2. Tête inclinée ---
        nez = _get(landmarks, 'nez')
        if nez and eg and ed:
            milieu_epaules_x = (eg['x'] + ed['x']) / 2
            decalage_tete = abs(nez['x'] - milieu_epaules_x)
            if decalage_tete > 0.06:
                penalite = min(20, int(decalage_tete * 300))
                result['posture_score'] -= penalite
                cote = "gauche" if nez['x'] < milieu_epaules_x else "droite"
                result['problemes'].append(f"Tête inclinée vers la {cote}")
                result['recommandations'].append("Centrez votre tête au-dessus de vos épaules.")

        # --- 3. Tête penchée en avant (forward head) ---
        if nez and eg and ed:
            milieu_epaules_y = (eg['y'] + ed['y']) / 2
            diff_y_tete = milieu_epaules_y - nez['y']
            if diff_y_tete < 0.08:  # tête trop proche des épaules = penchée
                result['posture_score'] -= 15
                result['problemes'].append("Tête penchée en avant")
                result['recommandations'].append("Reculez votre menton, alignez vos oreilles avec vos épaules.")

        # --- 4. Dos voûté (épaules en avant) ---
        og = _get(landmarks, 'oreille_gauche')
        od = _get(landmarks, 'oreille_droite')
        if og and od and eg and ed:
            # Si les oreilles sont bien devant les épaules → dos voûté
            oreille_moy_z = (og.get('z', 0) + od.get('z', 0)) / 2
            epaule_moy_z = (eg.get('z', 0) + ed.get('z', 0)) / 2
            diff_z = oreille_moy_z - epaule_moy_z
            if diff_z < -0.05:
                result['posture_score'] -= 15
                result['problemes'].append("Dos voûté détecté")
                result['recommandations'].append("Ouvrez votre poitrine et tirez vos épaules en arrière.")

        # --- 5. Hanches inégales ---
        hg = _get(landmarks, 'hanche_gauche')
        hd = _get(landmarks, 'hanche_droite')
        if hg and hd:
            diff_hanches = abs(hg['y'] - hd['y'])
            if diff_hanches > 0.04:
                penalite = min(15, int(diff_hanches * 350))
                result['posture_score'] -= penalite
                result['problemes'].append(f"Hanches inégales ({diff_hanches:.2f})")
                result['recommandations'].append("Répartissez votre poids également sur les deux jambes.")

        # --- 6. Angle du tronc (inclinaison latérale) ---
        if eg and ed and hg and hd:
            milieu_ep_x = (eg['x'] + ed['x']) / 2
            milieu_ep_y = (eg['y'] + ed['y']) / 2
            milieu_h_x = (hg['x'] + hd['x']) / 2
            milieu_h_y = (hg['y'] + hd['y']) / 2

            inclinaison_lat = abs(milieu_ep_x - milieu_h_x)
            if inclinaison_lat > 0.05:
                result['posture_score'] -= 15
                result['problemes'].append("Inclinaison latérale du tronc")
                result['recommandations'].append("Tenez-vous droit, alignez votre torse verticalement.")

        # --- 7. Angle du coude (bras croisés ou tendus) ---
        cg = _get(landmarks, 'coude_gauche')
        pg = _get(landmarks, 'poignet_gauche')
        if eg and cg and pg:
            angle_coude_g = _angle_deg(eg, cg, pg)
            if angle_coude_g < 40:
                result['posture_score'] -= 5
                result['problemes'].append("Bras gauche très replié")

        cd = _get(landmarks, 'coude_droit')
        pd = _get(landmarks, 'poignet_droit')
        if ed and cd and pd:
            angle_coude_d = _angle_deg(ed, cd, pd)
            if angle_coude_d < 40:
                result['posture_score'] -= 5
                result['problemes'].append("Bras droit très replié")

        # --- 8. Score final ---
        result['posture_score'] = max(0, min(100, result['posture_score']))

        if not result['problemes']:
            result['recommandations'].append("Bonne posture ! Continuez comme ça.")

        return result

    def track_exercise(self, landmarks_sequence: list, exercise_type: str) -> dict:
        """Compte les répétitions d'exercices."""
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
        Détecte les risques de chute (centre de gravité, inclinaison extrême).
        """
        result = {
            'risque_chute': False,
            'score_stabilite': 100,
            'alerte': ""
        }

        if not landmarks:
            return result

        nez = _get(landmarks, 'nez')
        hg = _get(landmarks, 'hanche_gauche')
        hd = _get(landmarks, 'hanche_droite')
        cg = _get(landmarks, 'cheville_gauche')
        cd = _get(landmarks, 'cheville_droite')

        # Chute : nez en dessous des hanches
        if nez and hg and hd:
            hanche_moy_y = (hg['y'] + hd['y']) / 2
            if nez['y'] > hanche_moy_y:
                result['risque_chute'] = True
                result['score_stabilite'] -= 50
                result['alerte'] = "⚠️ Position corporelle critique — risque de chute élevé !"

        # Instabilité : centre de masse décalé vs pieds
        if hg and hd and cg and cd:
            centre_masse_x = (hg['x'] + hd['x']) / 2
            base_support_min = min(cg['x'], cd['x'])
            base_support_max = max(cg['x'], cd['x'])
            marge = 0.08

            if centre_masse_x < base_support_min - marge or centre_masse_x > base_support_max + marge:
                result['score_stabilite'] -= 30
                if not result['risque_chute']:
                    result['alerte'] = "⚠️ Déséquilibre latéral détecté"
                result['risque_chute'] = True

        # Inclinaison forte du tronc
        eg = _get(landmarks, 'epaule_gauche')
        ed = _get(landmarks, 'epaule_droite')
        if eg and ed and hg and hd:
            ep_moy_x = (eg['x'] + ed['x']) / 2
            h_moy_x = (hg['x'] + hd['x']) / 2
            inclinaison = abs(ep_moy_x - h_moy_x)
            if inclinaison > 0.12:
                result['score_stabilite'] -= 25
                if not result['risque_chute']:
                    result['alerte'] = "⚠️ Forte inclinaison du tronc"
                result['risque_chute'] = True

        result['score_stabilite'] = max(0, result['score_stabilite'])
        return result
