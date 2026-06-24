import os
import cv2
import numpy as np

mask_dir = "data/masks"

values = set()

for f in os.listdir(mask_dir):
    path = os.path.join(mask_dir, f)
    mask = cv2.imread(path, 0)

    if mask is None:
        continue

    unique = np.unique(mask)
    for v in unique:
        values.add(int(v))

print("Unique values in dataset:", sorted(values))