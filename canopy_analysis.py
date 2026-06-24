# canopy_analysis.py
import numpy as np
from PIL import Image


def calculate_canopy_coverage(mask_path: str) -> dict:
    """
    Calculate canopy coverage percentage from a segmentation mask.
    Uses PIL instead of cv2 to avoid GUI dependency issues.
    
    Args:
        mask_path: Path to the output mask PNG
    Returns:
        dict with coverage_percent, white_pixels, total_pixels
    """
    try:
        mask = Image.open(mask_path).convert('L')  # Grayscale
        mask_array = np.array(mask)
        
        # Threshold: pixels > 127 are considered canopy (white)
        white_pixels = int(np.sum(mask_array > 127))
        total_pixels = int(mask_array.size)
        coverage_percent = round((white_pixels / total_pixels) * 100, 2)
        
        return {
            'coverage_percent': coverage_percent,
            'white_pixels': white_pixels,
            'total_pixels': total_pixels,
            'status': 'success'
        }
    except Exception as e:
        return {
            'coverage_percent': 0.0,
            'white_pixels': 0,
            'total_pixels': 0,
            'status': f'error: {str(e)}'
        }


def interpret_canopy(coverage_percent: float) -> dict:
    """Return canopy health interpretation and advice."""
    if coverage_percent < 15:
        return {
            'label': 'Sparse Canopy',
            'color': '#FF6B6B',
            'advice': 'Very low canopy density. Check for leaf drop, disease pressure, or poor vine establishment.',
            'icon': '⚠️'
        }
    elif coverage_percent < 35:
        return {
            'label': 'Moderate Canopy',
            'color': '#FFA500',
            'advice': 'Acceptable canopy density. Monitor growth and ensure adequate nutrition and water.',
            'icon': '📊'
        }
    elif coverage_percent < 60:
        return {
            'label': 'Good Canopy',
            'color': '#21C55D',
            'advice': 'Healthy canopy density. Good balance of leaf area for photosynthesis and air circulation.',
            'icon': '✅'
        }
    else:
        return {
            'label': 'Dense Canopy',
            'color': '#3B82F6',
            'advice': 'Very high canopy density. Consider summer pruning to improve airflow and reduce disease risk.',
            'icon': '🌿'
        }