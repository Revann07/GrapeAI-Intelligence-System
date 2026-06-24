import os
import cv2
import numpy as np

mask_dir = "data/masks"

for f in os.listdir(mask_dir):
    path = os.path.join(mask_dir, f)
    mask = cv2.imread(path, 0)

    if mask is None:
        continue

    vals = np.unique(mask)

    if 2 in vals:
        print("Mask with value 2:", f, vals)