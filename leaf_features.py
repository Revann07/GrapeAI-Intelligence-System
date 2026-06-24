# leaf_features.py
# Extracts quantitative leaf health metrics from RGB image

import numpy as np
from PIL import Image


def extract_leaf_features(image: Image.Image) -> dict:
    """
    Extracts 8 interpretable leaf health indicators.
    These metrics are used alongside CNN predictions for robust analysis.
    """
    img = np.array(image.convert('RGB').resize((256, 256))).astype(float)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    eps = 1e-6

    # 1. Vegetation Index (proxy NDVI using visible channels)
    ndvi = float(((g - r) / (g + r + eps)).mean())
    ndvi_norm = round(max(0.0, min(1.0, (ndvi + 1) / 2)), 3)

    # 2. Lesion density — dark abnormal spots
    gray = 0.299*r + 0.587*g + 0.114*b
    lesion_mask = (gray < 75) & (g < 80)
    lesion_pct = round(float(lesion_mask.mean()) * 100, 1)

    # 3. Yellowing index — chlorosis
    yellow_mask = (r > 150) & (g > 120) & (b < 85) & (r > g * 1.05)
    yellow_pct = round(float(yellow_mask.mean()) * 100, 1)

    # 4. Necrosis / browning
    brown_mask = (r > 100) & (r < 180) & (g < 85) & (b < 65)
    necrosis_pct = round(float(brown_mask.mean()) * 100, 1)

    # 5. Chlorophyll proxy — deep green content
    green_mask = (g > r * 1.15) & (g > b * 1.1) & (g > 70)
    chlorophyll = round(float(green_mask.mean()) * 100, 1)

    # 6. Leaf water content proxy — blue channel relative to green
    water_index = round(float((b / (g + eps)).mean()), 3)

    # 7. Texture roughness — local standard deviation
    from scipy.ndimage import uniform_filter
    gray_img = gray.astype(np.float32)
    mean_sq = uniform_filter(gray_img**2, size=5)
    sq_mean = uniform_filter(gray_img, size=5)**2
    texture = round(float(np.sqrt(np.maximum(mean_sq - sq_mean, 0)).mean()), 2)

    # 8. Overall health score (0–100)
    health = round(
        chlorophyll * 0.4 +
        (1 - necrosis_pct/100) * 30 +
        (1 - lesion_pct/100) * 20 +
        ndvi_norm * 10,
        1
    )
    health = min(max(health, 0), 100)

    return {
        'Vegetation Index':    ndvi_norm,
        'Chlorophyll Content': f'{chlorophyll}%',
        'Lesion Density':      f'{lesion_pct}%',
        'Yellowing (Chlorosis)': f'{yellow_pct}%',
        'Necrosis Area':       f'{necrosis_pct}%',
        'Water Content Index': water_index,
        'Texture Roughness':   texture,
        'Overall Health Score': f'{health}/100',
    }