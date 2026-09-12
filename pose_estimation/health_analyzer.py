"""
Analyseur biomécanique avancé de posture.
Utilise les angles articulaires réels (atan2, vecteurs 3D)
pour une détection précise des anomalies posturales.
Basé sur les standards cliniques de physiothérapie.
"""
import math


# ── Outils géométriques ─────────────────────────────────────

def _pt(landmarks, nom):
    """Récupère un landmark par nom, retourne None si absent."""
    return next((lm for lm in landmarks if lm['nom'] == nom), None)


def _angle_3pts(a, b, c):
    """Angle en degrés au sommet B entre les segments BA et BC."""
    ba = (a['x'] - b['x'], a['y'] - b['y'])
    bc = (c['x'] - b['x'], c['y'] - b['y'])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = (math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2)) + 1e-8
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))


def _angle_vertical(top, bottom):
    """Angle d'inclinaison par rapport à la verticale (0° = parfaitement droit)."""
    dx = top['x'] - bottom['x']
    dy = top['y'] - bottom['y']
    return abs(math.degrees(math.atan2(dx, -dy)))  # négatif car y augmente vers le bas


def _midpoint(a, b):
    """Point milieu entre deux landmarks."""
    return {
        'x': (a['x'] + b['x']) / 2,
        'y': (a['y'] + b['y']) / 2,
        'z': (a.get('z', 0) + b.get('z', 0)) / 2,
    }


def _distance(a, b):
    """Distance euclidienne 2D entre deux points."""
    return math.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)


# ── Analyseur principal ─────────────────────────────────────

class HealthAnalyzer:
    """
    Analyseur biomécanique avancé — 12 vérifications posturales
    basées sur les angles articulaires et les positions relatives.
    """

    def analyze_posture(self, landmarks: list) -> dict:
        """
        Analyse complète de la posture avec angles biomécaniques.

        Returns:
            dict: {posture_score (0-100), problemes, recommandations, details}
        """
        result = {
            'posture_score': 100,
            'problemes': [],
            'recommandations': [],
            'details': {}
        }

        if not landmarks or len(landmarks) < 25:
            return result

        checks = [
            self._check_epaules,
            self._check_tete_laterale,
            self._check_tete_avant,
            self._check_cou,
            self._check_dos_voute,
            self._check_hanches,
            self._check_tronc_lateral,
            self._check_tronc_frontal,
            self._check_genoux,
            self._check_symetrie_bras,
            self._check_epaules_relevees,
            self._check_position_globale,
        ]

        for check in checks:
            try:
                check(landmarks, result)
            except Exception:
                pass  # Ne pas crasher sur un check individuel

        result['posture_score'] = max(0, min(100, result['posture_score']))

        if not result['problemes']:
            result['recommandations'] = ["✅ Excellente posture ! Continuez."]

        return result

    # ── 1. Épaules inégales (roulis) ──────────────────────

    def _check_epaules(self, lm, r):
        eg = _pt(lm, 'epaule_gauche')
        ed = _pt(lm, 'epaule_droite')
        if not (eg and ed):
            return

        # Angle d'inclinaison de la ligne des épaules
        dx = ed['x'] - eg['x']
        dy = ed['y'] - eg['y']
        angle = abs(math.degrees(math.atan2(dy, dx)))
        r['details']['angle_epaules'] = round(angle, 1)

        # > 5° d'inclinaison = problème
        deviation = abs(angle - 180) if angle > 90 else angle
        if deviation > 5:
            penalite = min(20, int(deviation * 3))
            r['posture_score'] -= penalite
            cote = "gauche plus haute" if eg['y'] < ed['y'] else "droite plus haute"
            r['problemes'].append(f"Épaules inclinées ({deviation:.0f}°, {cote})")
            r['recommandations'].append("Alignez vos épaules horizontalement.")

    # ── 2. Tête inclinée latéralement ─────────────────────

    def _check_tete_laterale(self, lm, r):
        nez = _pt(lm, 'nez')
        eg = _pt(lm, 'epaule_gauche')
        ed = _pt(lm, 'epaule_droite')
        if not (nez and eg and ed):
            return

        milieu = _midpoint(eg, ed)
        decalage = nez['x'] - milieu['x']
        largeur_epaules = abs(ed['x'] - eg['x']) + 1e-6
        ratio = abs(decalage) / largeur_epaules
        r['details']['decalage_tete_lateral'] = round(ratio, 3)

        if ratio > 0.15:
            penalite = min(20, int(ratio * 100))
            r['posture_score'] -= penalite
            cote = "gauche" if decalage < 0 else "droite"
            r['problemes'].append(f"Tête inclinée vers la {cote} ({ratio:.0%})")
            r['recommandations'].append("Centrez votre tête au-dessus de vos épaules.")

    # ── 3. Tête penchée en avant (forward head) ───────────

    def _check_tete_avant(self, lm, r):
        og = _pt(lm, 'oreille_gauche')
        od = _pt(lm, 'oreille_droite')
        eg = _pt(lm, 'epaule_gauche')
        ed = _pt(lm, 'epaule_droite')
        if not (og and od and eg and ed):
            return

        oreille = _midpoint(og, od)
        epaule = _midpoint(eg, ed)

        # En vue profil, les oreilles doivent être au-dessus des épaules
        # Profondeur Z : oreille devant épaule = tête en avant
        diff_z = oreille.get('z', 0) - epaule.get('z', 0)
        r['details']['forward_head_z'] = round(diff_z, 3)

        if diff_z < -0.04:
            penalite = min(20, int(abs(diff_z) * 300))
            r['posture_score'] -= penalite
            r['problemes'].append(f"Tête projetée en avant (Δz={diff_z:.2f})")
            r['recommandations'].append("Rentrez le menton et reculez la tête.")

    # ── 4. Angle du cou ───────────────────────────────────

    def _check_cou(self, lm, r):
        nez = _pt(lm, 'nez')
        eg = _pt(lm, 'epaule_gauche')
        ed = _pt(lm, 'epaule_droite')
        hg = _pt(lm, 'hanche_gauche')
        hd = _pt(lm, 'hanche_droite')
        if not (nez and eg and ed and hg and hd):
            return

        epaule = _midpoint(eg, ed)
        hanche = _midpoint(hg, hd)

        # Angle nez-épaule par rapport à épaule-hanche
        angle_cou = _angle_3pts(nez, epaule, hanche)
        r['details']['angle_cou'] = round(angle_cou, 1)

        # Angle normal du cou : 160-180°. < 150° = problème
        if angle_cou < 150:
            penalite = min(15, int((150 - angle_cou) * 1.5))
            r['posture_score'] -= penalite
            r['problemes'].append(f"Cou fléchi ({angle_cou:.0f}°, normal > 150°)")
            r['recommandations'].append("Allongez votre cou, gardez la tête haute.")

    # ── 5. Dos voûté (cyphose) ────────────────────────────

    def _check_dos_voute(self, lm, r):
        eg = _pt(lm, 'epaule_gauche')
        ed = _pt(lm, 'epaule_droite')
        hg = _pt(lm, 'hanche_gauche')
        hd = _pt(lm, 'hanche_droite')
        if not (eg and ed and hg and hd):
            return

        epaule = _midpoint(eg, ed)
        hanche = _midpoint(hg, hd)

        # Angle d'inclinaison du tronc par rapport à la verticale
        angle_tronc = _angle_vertical(epaule, hanche)
        r['details']['angle_tronc_vertical'] = round(angle_tronc, 1)

        if angle_tronc > 12:
            penalite = min(20, int((angle_tronc - 12) * 3))
            r['posture_score'] -= penalite
            r['problemes'].append(f"Dos voûté ({angle_tronc:.0f}° d'inclinaison)")
            r['recommandations'].append("Redressez votre dos, ouvrez la poitrine.")

    # ── 6. Hanches inégales ───────────────────────────────

    def _check_hanches(self, lm, r):
        hg = _pt(lm, 'hanche_gauche')
        hd = _pt(lm, 'hanche_droite')
        if not (hg and hd):
            return

        diff = abs(hg['y'] - hd['y'])
        largeur = abs(hd['x'] - hg['x']) + 1e-6
        ratio = diff / largeur
        r['details']['asymetrie_hanches'] = round(ratio, 3)

        if ratio > 0.08:
            penalite = min(15, int(ratio * 150))
            r['posture_score'] -= penalite
            cote = "gauche haute" if hg['y'] < hd['y'] else "droite haute"
            r['problemes'].append(f"Hanches asymétriques ({cote})")
            r['recommandations'].append("Répartissez votre poids sur les deux pieds.")

    # ── 7. Inclinaison latérale du tronc ──────────────────

    def _check_tronc_lateral(self, lm, r):
        eg = _pt(lm, 'epaule_gauche')
        ed = _pt(lm, 'epaule_droite')
        hg = _pt(lm, 'hanche_gauche')
        hd = _pt(lm, 'hanche_droite')
        if not (eg and ed and hg and hd):
            return

        ep = _midpoint(eg, ed)
        ha = _midpoint(hg, hd)
        decalage = abs(ep['x'] - ha['x'])
        hauteur = abs(ep['y'] - ha['y']) + 1e-6
        ratio = decalage / hauteur
        r['details']['inclinaison_laterale'] = round(ratio, 3)

        if ratio > 0.10:
            penalite = min(15, int(ratio * 120))
            r['posture_score'] -= penalite
            cote = "gauche" if ep['x'] < ha['x'] else "droite"
            r['problemes'].append(f"Tronc penché vers la {cote}")
            r['recommandations'].append("Recentrez votre buste verticalement.")

    # ── 8. Inclinaison avant/arrière du tronc ─────────────

    def _check_tronc_frontal(self, lm, r):
        eg = _pt(lm, 'epaule_gauche')
        ed = _pt(lm, 'epaule_droite')
        hg = _pt(lm, 'hanche_gauche')
        hd = _pt(lm, 'hanche_droite')
        if not (eg and ed and hg and hd):
            return

        ep_z = (eg.get('z', 0) + ed.get('z', 0)) / 2
        ha_z = (hg.get('z', 0) + hd.get('z', 0)) / 2
        diff_z = ep_z - ha_z
        r['details']['inclinaison_avant_z'] = round(diff_z, 3)

        if diff_z < -0.06:
            r['posture_score'] -= 10
            r['problemes'].append("Buste penché en avant")
            r['recommandations'].append("Redressez-vous, ramenez vos épaules en arrière.")

    # ── 9. Genoux (hyperextension / flexion) ──────────────

    def _check_genoux(self, lm, r):
        for cote, h_nom, g_nom, c_nom in [
            ("gauche", "hanche_gauche", "genou_gauche", "cheville_gauche"),
            ("droit", "hanche_droite", "genou_droit", "cheville_droite"),
        ]:
            h = _pt(lm, h_nom)
            g = _pt(lm, g_nom)
            c = _pt(lm, c_nom)
            if not (h and g and c):
                continue

            angle = _angle_3pts(h, g, c)
            r['details'][f'angle_genou_{cote}'] = round(angle, 1)

            if angle < 140:
                r['posture_score'] -= 8
                r['problemes'].append(f"Genou {cote} fléchi ({angle:.0f}°)")
                r['recommandations'].append(f"Étendez votre genou {cote}.")
            elif angle > 185:
                r['posture_score'] -= 8
                r['problemes'].append(f"Hyperextension genou {cote} ({angle:.0f}°)")
                r['recommandations'].append(f"Évitez de verrouiller votre genou {cote}.")

    # ── 10. Symétrie des bras ─────────────────────────────

    def _check_symetrie_bras(self, lm, r):
        eg = _pt(lm, 'epaule_gauche')
        cg = _pt(lm, 'coude_gauche')
        pg = _pt(lm, 'poignet_gauche')
        ed = _pt(lm, 'epaule_droite')
        cd = _pt(lm, 'coude_droit')
        pd = _pt(lm, 'poignet_droit')

        if not (eg and cg and pg and ed and cd and pd):
            return

        angle_g = _angle_3pts(eg, cg, pg)
        angle_d = _angle_3pts(ed, cd, pd)
        diff = abs(angle_g - angle_d)
        r['details']['angle_coude_gauche'] = round(angle_g, 1)
        r['details']['angle_coude_droit'] = round(angle_d, 1)
        r['details']['asymetrie_bras'] = round(diff, 1)

        if diff > 40:
            r['posture_score'] -= 5
            r['problemes'].append(f"Bras asymétriques (Δ{diff:.0f}°)")

    # ── 11. Épaules relevées (tension) ────────────────────

    def _check_epaules_relevees(self, lm, r):
        og = _pt(lm, 'oreille_gauche')
        od = _pt(lm, 'oreille_droite')
        eg = _pt(lm, 'epaule_gauche')
        ed = _pt(lm, 'epaule_droite')
        if not (og and od and eg and ed):
            return

        # Distance oreille-épaule (petite = épaules relevées/tension)
        dist_g = abs(og['y'] - eg['y'])
        dist_d = abs(od['y'] - ed['y'])
        dist_moy = (dist_g + dist_d) / 2
        r['details']['dist_oreille_epaule'] = round(dist_moy, 3)

        if dist_moy < 0.06:
            r['posture_score'] -= 10
            r['problemes'].append("Épaules relevées (tension musculaire)")
            r['recommandations'].append("Relâchez vos épaules, laissez-les descendre.")

    # ── 12. Position globale (stabilité) ──────────────────

    def _check_position_globale(self, lm, r):
        nez = _pt(lm, 'nez')
        hg = _pt(lm, 'hanche_gauche')
        hd = _pt(lm, 'hanche_droite')
        cg = _pt(lm, 'cheville_gauche')
        cd = _pt(lm, 'cheville_droite')

        if not (nez and hg and hd):
            return

        hanche = _midpoint(hg, hd)

        # Tête sous les hanches = chute ou position anormale
        if nez['y'] > hanche['y']:
            r['posture_score'] -= 25
            r['problemes'].append("Position anormale — tête sous les hanches")
            r['recommandations'].append("Redressez-vous immédiatement.")

    # ── Suivi d'exercices ─────────────────────────────────

    def track_exercise(self, landmarks_sequence: list, exercise_type: str) -> dict:
        """Comptage de répétitions (placeholder)."""
        return {
            'repetitions': 0,
            'angle_moyen': 0.0,
            'qualite_mouvement': "Non évalué"
        }

    # ── Détection de chute ────────────────────────────────

    def detect_fall_risk(self, landmarks: list) -> dict:
        """Analyse avancée du risque de chute."""
        result = {
            'risque_chute': False,
            'score_stabilite': 100,
            'alerte': ""
        }

        if not landmarks:
            return result

        nez = _pt(landmarks, 'nez')
        hg = _pt(landmarks, 'hanche_gauche')
        hd = _pt(landmarks, 'hanche_droite')
        cg = _pt(landmarks, 'cheville_gauche')
        cd = _pt(landmarks, 'cheville_droite')
        eg = _pt(landmarks, 'epaule_gauche')
        ed = _pt(landmarks, 'epaule_droite')

        # 1. Tête sous les hanches
        if nez and hg and hd:
            hanche_y = (hg['y'] + hd['y']) / 2
            if nez['y'] > hanche_y:
                result['risque_chute'] = True
                result['score_stabilite'] -= 50
                result['alerte'] = "⚠️ Position critique — risque de chute élevé !"

        # 2. Centre de masse hors de la base de support
        if hg and hd and cg and cd:
            cm_x = (hg['x'] + hd['x']) / 2
            base_min = min(cg['x'], cd['x']) - 0.05
            base_max = max(cg['x'], cd['x']) + 0.05

            if cm_x < base_min or cm_x > base_max:
                result['score_stabilite'] -= 30
                result['risque_chute'] = True
                if not result['alerte']:
                    result['alerte'] = "⚠️ Déséquilibre — centre de gravité hors appui"

        # 3. Inclinaison extrême du tronc
        if eg and ed and hg and hd:
            ep = _midpoint(eg, ed)
            ha = _midpoint(hg, hd)
            angle = _angle_vertical(ep, ha)

            if angle > 25:
                result['score_stabilite'] -= 25
                result['risque_chute'] = True
                if not result['alerte']:
                    result['alerte'] = f"⚠️ Inclinaison extrême ({angle:.0f}°)"

        # 4. Chevilles trop rapprochées (base instable)
        if cg and cd:
            ecart = abs(cg['x'] - cd['x'])
            if ecart < 0.05:
                result['score_stabilite'] -= 15
                if not result['alerte']:
                    result['alerte'] = "⚠️ Base d'appui trop étroite"

        result['score_stabilite'] = max(0, result['score_stabilite'])
        return result
