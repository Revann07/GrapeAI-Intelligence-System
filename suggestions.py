# suggestions.py

SUGGESTIONS = {
    'Grape___Black_rot': {
        'severity_note': 'Fungal disease caused by Guignardia bidwellii. Can cause up to 80% yield loss.',
        'treatment': [
            'Apply fungicide (Mancozeb or Captan) every 7–10 days during wet seasons.',
            'Remove and destroy all mummified berries and infected canes.',
            'Apply Myclobutanil or Tebuconazole as curative spray if infection is active.',
            'Avoid overhead irrigation; use drip irrigation to keep foliage dry.',
        ],
        'pruning': [
            'Prune out all infected canes at least 15 cm below visible lesions.',
            'Improve canopy air circulation by thinning lateral shoots.',
            'Open up the vine canopy to reduce humidity and fungal spread.',
            'Disinfect pruning tools with 10% bleach solution between cuts.',
        ],
        'prevention': [
            'Apply dormant copper spray before bud break.',
            'Train vines to allow maximum sunlight penetration.',
            'Monitor from pre-bloom through harvest; highest risk period is bloom to 4 weeks post-bloom.',
        ],
        'urgency': 'HIGH',
        'icon': '🔴'
    },
    'Grape___Esca_(Black_Measles)': {
        'severity_note': 'Complex wood disease caused by multiple fungi (Phaeomoniella, Phaeoacremonium). Affects vascular tissue.',
        'treatment': [
            'No fully curative chemical treatment exists; management focuses on slowing spread.',
            'Apply sodium arsenite (where legally permitted) as a trunk wound protectant.',
            'Use Trichoderma-based biological agents on pruning wounds.',
            'Apply wound sealant paste immediately after all pruning cuts.',
        ],
        'pruning': [
            'Remove severely affected cordons or trunks back to healthy wood.',
            'Make clean cuts and immediately seal with fungicidal wound paint.',
            'Consider trunk renewal — retrain young suckers as replacement trunks.',
            'Avoid large pruning cuts; minimize wound size wherever possible.',
        ],
        'prevention': [
            'Protect all pruning wounds with Botrytis/Eutypa fungicide within 24 hours.',
            'Prune during dry weather to minimize spore germination on fresh cuts.',
            'Avoid pruning when rain is forecast within 48 hours.',
        ],
        'urgency': 'HIGH',
        'icon': '🔴'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'severity_note': 'Caused by Pseudocercospora vitis. Primarily affects leaves; reduces photosynthesis and weakens vine.',
        'treatment': [
            'Spray with copper-based fungicide (Bordeaux mixture) at first sign of lesions.',
            'Apply Chlorothalonil or Mancozeb every 10–14 days during humid weather.',
            'Remove heavily infected leaves to reduce inoculum load.',
            'Ensure adequate potassium nutrition to improve leaf resistance.',
        ],
        'pruning': [
            'Remove and dispose of infected leaves (do not compost).',
            'Thin the canopy to improve airflow between leaves.',
            'Avoid excessive nitrogen fertilization which promotes dense, susceptible growth.',
            'After harvest, remove all fallen infected leaf debris from the vineyard floor.',
        ],
        'prevention': [
            'Apply protectant fungicide spray before wet/humid periods.',
            'Maintain balanced vine nutrition — avoid excess nitrogen.',
            'Scout weekly from veraison onwards for early detection.',
        ],
        'urgency': 'MEDIUM',
        'icon': '🟡'
    },
    'Grape___healthy': {
        'severity_note': 'Vine appears healthy. No disease symptoms detected.',
        'treatment': [
            'Continue current management practices — they are working well.',
            'Maintain regular preventive fungicide schedule during high-risk periods.',
            'Monitor weekly for early signs of disease or pest pressure.',
        ],
        'pruning': [
            'Perform routine dormant pruning to maintain balanced vine structure.',
            'Remove crossing or crowding canes to maintain good air circulation.',
            'Maintain the desired training system (VSP, GDC, Scott Henry, etc.).',
            'Summer prune to manage excessive canopy density after fruit set.',
        ],
        'prevention': [
            'Apply dormant copper spray before bud break as standard preventive measure.',
            'Keep vineyard floor mowed to reduce humidity at vine base.',
            'Ensure adequate nutrition through petiole testing and soil analysis.',
        ],
        'urgency': 'LOW',
        'icon': '🟢'
    }
}


def get_suggestions(class_name: str) -> dict:
    """Return full suggestion dict for a given disease class name."""
    return SUGGESTIONS.get(class_name, SUGGESTIONS['Grape___healthy'])


def get_urgency_color(urgency: str) -> str:
    colors = {
        'HIGH': '#FF4B4B',
        'MEDIUM': '#FFA500',
        'LOW': '#21C55D',
    }
    return colors.get(urgency, '#888888')