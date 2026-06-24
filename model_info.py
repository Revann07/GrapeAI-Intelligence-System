# model_info.py
MODEL_REGISTRY = {
    "GrapeCanopyNet (U-Net)": {
        "Architecture":    "Encoder-Decoder with skip connections",
        "Backbone":        "Custom conv blocks (scratch)",
        "Training images": "587 annotated grape canopy images",
        "Epochs":          "30",
        "Optimizer":       "Adam  lr=1e-4, weight_decay=1e-5",
        "Loss function":   "Binary Cross-Entropy + Dice Loss",
        "Input size":      "572 × 572 px",
        "Output":          "Binary segmentation mask",
        "Checkpoint":      "checkpoint_epoch30.pth",
    },
    "GrapeLeafCNN (Disease)": {
        "Architecture":    "EfficientNet-B4 + 3-layer custom head",
        "Strategy":        "Transfer learning — ImageNet init, fine-tuned head",
        "Dataset":         "PlantVillage grape subset (4 classes, 4,062 images)",
        "Classes":         "Black Rot | Esca | Leaf Blight | Healthy",
        "Head layers":     "FC(1792→512) → BN → FC(512→128) → FC(128→4)",
        "Regularization":  "Dropout(0.3) + Dropout(0.2) + BatchNorm",
        "Input size":      "380 × 380 px",
        "Framework":       "PyTorch 2.x + torchvision",
    },
}

DATASET_STATS = {
    "Canopy Dataset": {
        "Total":       587,
        "Train":       470,
        "Validation":  117,
        "Annotations": "Manual polygon (LabelMe)",
        "Augment":     "Flip, rotate±30°, brightness, elastic warp",
    },
    "PlantVillage (Grape)": {
        "Total":       4062,
        "Black Rot":   1180,
        "Esca":        1383,
        "Leaf Blight": 1076,
        "Healthy":     423,
        "Split":       "80 / 20 train / val",
    },
}

def print_model_info():
    w = 58
    print("\n" + "=" * w)
    print("  GrapeAI — Model Registry".center(w))
    print("=" * w)
    for model, info in MODEL_REGISTRY.items():
        print(f"\n  [{model}]")
        for k, v in info.items():
            print(f"    {k:<20}: {v}")
    print("\n" + "-" * w)
    print("  Dataset Statistics".center(w))
    print("-" * w)
    for ds, info in DATASET_STATS.items():
        print(f"\n  [{ds}]")
        for k, v in info.items():
            print(f"    {k:<16}: {v}")
    print("=" * w + "\n")