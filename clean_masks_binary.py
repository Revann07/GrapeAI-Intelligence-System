import os
import cv2

mask_dir = "data/masks"

for filename in os.listdir(mask_dir):

    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    path = os.path.join(mask_dir, filename)
    mask = cv2.imread(path, 0)

    if mask is None:
        continue

    # Convert to strict binary
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    cv2.imwrite(path, mask)

    print(f"Cleaned {filename}")

print("All masks converted to binary (0 and 1).")