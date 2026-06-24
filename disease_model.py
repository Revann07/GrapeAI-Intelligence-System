# disease_model.py
# GrapeLeafCNN — Custom architecture with EfficientNet-B4 backbone
# Designed for 4-class grape disease classification

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
import numpy as np

DISEASE_CLASSES = [
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
]

DISPLAY_NAMES = {
    'Grape___Black_rot':                          'Grape Black Rot',
    'Grape___Esca_(Black_Measles)':               'Grape Esca (Black Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape Leaf Blight',
    'Grape___healthy':                            'Healthy Grape Leaf',
}

transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class GrapeLeafCNN(nn.Module):
    """
    Custom CNN for grape disease classification.
    Architecture: EfficientNet-B4 backbone + domain-specific 3-layer head.
    Strategy: Transfer learning — backbone frozen, head fine-tuned on PlantVillage.
    """
    def __init__(self, num_classes=4, freeze_backbone=True):
        super().__init__()
        base = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.feature_extractor = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Freeze early layers (blocks 0-4), fine-tune deeper layers
        if freeze_backbone:
            for i, layer in enumerate(self.feature_extractor):
                if i < 5:
                    for param in layer.parameters():
                        param.requires_grad = False

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1792, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.pool(self.feature_extractor(x)).flatten(1)
        return self.classifier(x)


def load_disease_model(model_path=None):
    model = GrapeLeafCNN(num_classes=4, freeze_backbone=True)
    if model_path:
        try:
            state = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state)
            print(f"[GrapeLeafCNN] Loaded custom weights from {model_path}")
        except Exception as e:
            print(f"[GrapeLeafCNN] Custom weights not found ({e}). Using ImageNet init.")
    model.eval()
    return model


def _color_heuristic(image: Image.Image) -> np.ndarray:
    """Vegetation-based heuristic to assist classification."""
    img = np.array(image.convert('RGB').resize((224, 224))).astype(float)
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    brown  = ((r > 100) & (g < 80)  & (b < 60)).mean()
    yellow = ((r > 150) & (g > 120) & (b < 80)).mean()
    edge   = np.concatenate([img[:20,:,:3].reshape(-1,3),
                              img[-20:,:,:3].reshape(-1,3)])
    edge_brown = ((edge[:,0] > 120) & (edge[:,1] < 90)).mean()
    green  = ((g > r * 1.1) & (g > b * 1.1) & (g > 60)).mean()

    h = np.array([brown*3.0, yellow*2.5, edge_brown*2.0, green*1.5])
    return h / (h.sum() + 1e-6)


def predict_disease(model, image: Image.Image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    heuristic   = _color_heuristic(image)
    final_probs = 0.55 * probs + 0.45 * heuristic
    final_probs = final_probs / final_probs.sum()

    idx          = int(np.argmax(final_probs))
    class_name   = DISEASE_CLASSES[idx]
    display_name = DISPLAY_NAMES[class_name]
    confidence   = float(final_probs[idx])
    all_probs    = {DISPLAY_NAMES[c]: float(final_probs[i])
                    for i, c in enumerate(DISEASE_CLASSES)}

    return class_name, display_name, confidence, all_probs