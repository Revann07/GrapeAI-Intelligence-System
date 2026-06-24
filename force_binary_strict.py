import os
import cv2

mask_dir = "data/masks"

for f in os.listdir(mask_dir):

    path = os.path.join(mask_dir, f)
    mask = cv2.imread(path, 0)

    if mask is None:
        continue

    mask[mask > 0] = 1

    cv2.imwrite(path, mask)

print("Done")