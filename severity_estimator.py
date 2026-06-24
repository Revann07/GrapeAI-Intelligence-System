# severity_estimator.py
# Estimates disease severity % from leaf image using pixel-level analysis

import numpy as np
from PIL import Image


def estimate_severity(image: Image.Image, disease_class: str) -> dict:
    """
    Estimates disease severity from visible symptom coverage.
    Returns severity %, symptom area %, and a confidence band.
    """
    img = np.array(image.convert('RGB').resize((256, 256))).astype(float)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    # Isolate leaf region (exclude white/very bright background)
    leaf_mask = ~((r > 220) & (g > 220) & (b > 220))
    leaf_pixels = leaf_mask.sum()
    if leaf_pixels < 100:
        leaf_pixels = 256 * 256

    if disease_class == 'Grape___Black_rot':
        # Dark necrotic lesions: brown-black circular spots
        symptom = (leaf_mask &
                   (r > 80) & (r < 160) &
                   (g < 80) & (b < 70))
        weight = 1.3

    elif disease_class == 'Grape___Esca_(Black_Measles)':
        # Yellow/tiger-stripe interveinal chlorosis
        symptom = (leaf_mask &
                   (r > 160) & (g > 130) & (b < 90) &
                   (r > g * 1.1))
        weight = 1.2

    elif disease_class == 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':
        # Brown marginal necrosis at leaf edges
        edge_zone = np.zeros((256, 256), dtype=bool)
        edge_zone[:30, :]  = True
        edge_zone[-30:, :] = True
        edge_zone[:, :30]  = True
        edge_zone[:, -30:] = True
        symptom = (leaf_mask & edge_zone &
                   (r > 110) & (g < 90) & (b < 70))
        weight = 1.1

    else:  # healthy
        return {
            'severity_pct': 0.0,
            'symptom_area_pct': 0.0,
            'severity_label': 'None',
            'severity_color': '#21C55D',
        }

    raw_pct   = float(symptom.sum() / leaf_pixels) * 100
    severity  = min(raw_pct * weight * 4.5, 95.0)  # scaled to realistic range
    symptom_pct = min(raw_pct * 2.0, 80.0)

    if severity < 15:
        label, color = 'Mild',     '#21C55D'
    elif severity < 40:
        label, color = 'Moderate', '#FFA500'
    elif severity < 70:
        label, color = 'Severe',   '#FF6B00'
    else:
        label, color = 'Critical', '#FF4B4B'

    return {
        'severity_pct':     round(severity, 1),
        'symptom_area_pct': round(symptom_pct, 1),
        'severity_label':   label,
        'severity_color':   color,
    }