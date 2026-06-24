import os
import cv2
import numpy as np

mask_dir = "data/masks"

for f in os.listdir(mask_dir):
    path = os.path.join(mask_dir, f)

    mask = cv2.imread(path)

    if mask is None:
        continue

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = np.where(gray > 0, 1, 0).astype(np.uint8)

    cv2.imwrite(path, binary)

print("Masks converted correctly to 0 and 1")