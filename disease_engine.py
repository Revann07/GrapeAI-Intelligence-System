"""
disease_engine.py
Unified Disease Decision Engine for GrapeAI Intelligence System.
"""

import re
import numpy as np
from PIL import Image

LEAF_DISEASE_NAMES = [
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy'
]

DISPLAY_NAMES = {
    'Grape___Black_rot': 'Grape Black Rot',
    'Grape___Esca_(Black_Measles)': 'Grape Esca (Black Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape Leaf Blight',
    'Grape___healthy': 'Healthy Grape Leaf'
}

def detect_image_type(image: Image.Image) -> dict:
    img_array = np.array(image.convert('RGB').resize((64, 64))).astype(float)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    total_pixels = 64 * 64
    
    green_pixels = (g > r * 1.1) & (g > b * 1.1) & (g > 50)
    leaf_ratio = np.sum(green_pixels) / total_pixels
    
    purple_pixels = (r > g) & (b > g) & ((r + b) > 100)
    dark_shadows = (r < 50) & (g < 50) & (b < 50)
    bunch_ratio = (np.sum(purple_pixels) + np.sum(dark_shadows)) / total_pixels
    
    if leaf_ratio > 0.35 and leaf_ratio > bunch_ratio * 1.5:
        confidence = min(0.6 + (leaf_ratio * 0.4), 0.99)
        return {"type": "leaf", "confidence": float(confidence)}
    elif bunch_ratio > 0.15:
        confidence = min(0.6 + (bunch_ratio * 0.4), 0.98)
        return {"type": "bunch", "confidence": float(confidence)}
    else:
        return {"type": "ambiguous", "confidence": 0.55}

def parse_gemini_output(text: str) -> dict:
    parsed = {
        "disease_name": None, "confidence_label": None, "severity_label": None,
        "severity_pct": None, "symptoms": "", "treatment": "", "prevention": ""
    }
    if not text or not isinstance(text, str): return parsed

    d_match = re.search(r'(?i)(?:disease|diagnosis|condition|detected)(?:\s*name)?\s*[:\-]\s*\*?\*?\s*([a-zA-Z0-9\s\-\_]+)', text)
    if d_match: parsed["disease_name"] = d_match.group(1).strip(" *_\n")

    c_match = re.search(r'(?i)confidence\s*[:\-]\s*\*?\*?\s*([0-9]+%?|[a-zA-Z]+)', text)
    if c_match: parsed["confidence_label"] = c_match.group(1).strip(" *_\n")

    s_match = re.search(r'(?i)severity\s*[:\-]\s*\*?\*?\s*(low|medium|high|severe|mild|moderate)', text)
    if s_match: parsed["severity_label"] = s_match.group(1).strip(" *_\n").capitalize()

    pct_match = re.search(r'(?i)severity.*?([0-9]{1,3})\s*%', text)
    if pct_match: parsed["severity_pct"] = int(pct_match.group(1))

    def extract_section(header_regex, stops_regex):
        pattern = re.compile(rf'(?i){header_regex}\s*[:\-]?\s*(.*?)(?={stops_regex}|$)', re.DOTALL)
        match = pattern.search(text)
        if match: return re.sub(r'^\*?\*?\s*', '', match.group(1).strip())
        return ""

    stops = r'\n\*?\*?(symptoms|treatment|prevention|recommendations|conclusion|severity|confidence|disease)'
    
    parsed["symptoms"] = extract_section(r'(?:symptoms|signs|indicators)', stops)
    parsed["treatment"] = extract_section(r'(?:treatment|action|cure|mitigation)', stops)
    parsed["prevention"] = extract_section(r'(?:prevention|control|management|proactive)', stops)
    return parsed

def _parse_confidence(conf_str) -> float:
    if not conf_str: return 0.85
    c = str(conf_str).lower()
    num = re.search(r'([0-9]+)', c)
    if num: return min(float(num.group(1)) / 100.0, 0.99)
    if "high" in c: return 0.90
    if "medium" in c or "moderate" in c: return 0.70
    if "low" in c: return 0.40
    return 0.85

def _blend_severity(gemini_label: str, pixel_sev: dict, gemini_pct: int = None) -> tuple:
    if pixel_sev is None: pixel_sev = {}
    base_pct = pixel_sev.get("severity_pct", 0)
    blended_pct = base_pct

    if gemini_pct is not None:
        blended_pct = max(base_pct, gemini_pct)
    else:
        g_label = (gemini_label or "").lower()
        if "high" in g_label or "severe" in g_label: blended_pct = max(base_pct, 85)
        elif "medium" in g_label or "moderate" in g_label: blended_pct = max(base_pct, 45)
        elif "low" in g_label or "mild" in g_label: blended_pct = min(max(base_pct, 10), 25)

    if blended_pct >= 60: label, color, urgency = "High", "#ff4b4b", "HIGH"
    elif blended_pct >= 25: label, color, urgency = "Medium", "#ffa500", "MEDIUM"
    elif blended_pct > 0: label, color, urgency = "Low", "#21c55d", "LOW"
    else: label, color, urgency = "Healthy", "#21c55d", "LOW"

    severity_obj = {
        "severity_pct": int(blended_pct),
        "severity_label": label,
        "severity_color": color,
        "symptom_area_pct": pixel_sev.get("symptom_area_pct", 0),
        "source": "Canopy AI" if gemini_pct else ("Fusion Engine" if gemini_label else "Pixel Analysis")
    }
    return severity_obj, urgency

def resolve_diagnosis(img_type: dict, cnn_class: str, cnn_conf: float, cnn_probs: dict, gemini_parsed: dict, pixel_severity: dict) -> dict:
    itype = img_type.get("type", "ambiguous")
    g_disease = gemini_parsed.get("disease_name")
    g_conf = _parse_confidence(gemini_parsed.get("confidence_label"))
    g_sev_label = gemini_parsed.get("severity_label")
    g_sev_pct = gemini_parsed.get("severity_pct") 
    cnn_display = DISPLAY_NAMES.get(cnn_class, "Unknown")
    
    if itype == "bunch":
        disease = g_disease if g_disease else "Unknown Grape Condition"
        sev, urg = _blend_severity(g_sev_label, pixel_severity, g_sev_pct)
        return {"disease_name": disease, "confidence": g_conf, "probs": {"Canopy AI": g_conf, "Other": 1.0 - g_conf}, "severity": sev, "urgency": urg, "source": "gemini", "reason": "Bunch image detected. GrapeLeafCNN bypassed; relying on Canopy AI.", "is_leaf_disease": False}
        
    elif itype == "leaf":
        if cnn_conf < 0.65 and g_disease and g_conf > 0.70:
            sev, urg = _blend_severity(g_sev_label, pixel_severity, g_sev_pct)
            return {"disease_name": g_disease, "confidence": g_conf, "probs": {"Canopy AI Override": g_conf, f"CNN: {cnn_display}": cnn_conf}, "severity": sev, "urgency": urg, "source": "cnn+gemini", "reason": f"Leaf detected, but CNN confidence too low ({cnn_conf:.1%}). Canopy AI Override applied.", "is_leaf_disease": True}
        else:
            sev, urg = _blend_severity(None, pixel_severity) 
            return {"disease_name": cnn_display, "confidence": cnn_conf, "probs": cnn_probs, "severity": sev, "urgency": urg, "source": "cnn", "reason": "Leaf image detected. Primary GrapeLeafCNN trusted.", "is_leaf_disease": True}

    else:
        if g_disease and (g_conf > cnn_conf):
            sev, urg = _blend_severity(g_sev_label, pixel_severity, g_sev_pct)
            return {"disease_name": g_disease, "confidence": g_conf, "probs": {"Canopy AI": g_conf, f"CNN: {cnn_display}": cnn_conf}, "severity": sev, "urgency": urg, "source": "cnn+gemini", "reason": "Ambiguous image. Canopy AI yielded higher confidence; employing fusion.", "is_leaf_disease": False}
        else:
            sev, urg = _blend_severity(None, pixel_severity)
            return {"disease_name": cnn_display, "confidence": cnn_conf, "probs": cnn_probs, "severity": sev, "urgency": urg, "source": "cnn", "reason": "Ambiguous image. GrapeLeafCNN confidence prevailed.", "is_leaf_disease": True}

def get_source_badge(source: str) -> tuple:
    if source == "cnn":
        return ('<span style="background:#2e7d32;color:#fff;padding:3px 8px;border-radius:4px;font-size:0.7rem;font-weight:600">GrapeLeafCNN</span>', "GrapeLeafCNN")
    elif source == "gemini":
        return ('<span style="background:#1565c0;color:#fff;padding:3px 8px;border-radius:4px;font-size:0.7rem;font-weight:600">Canopy AI</span>', "Canopy AI")
    elif source == "cnn+gemini":
        return ('<span style="background:#6a1b9a;color:#fff;padding:3px 8px;border-radius:4px;font-size:0.7rem;font-weight:600">Fusion Engine</span>', "Fusion Engine")
    else:
        return ('<span style="background:#e65100;color:#fff;padding:3px 8px;border-radius:4px;font-size:0.7rem;font-weight:600">Uncertain</span>', "Uncertain")