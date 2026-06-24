import os
import cv2

img_dir = "data/imgs"
mask_dir = "data/masks"

for filename in os.listdir(img_dir):

    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(img_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    if not os.path.exists(mask_path):
        print(f"Mask missing for {filename}")
        continue

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if image is None or mask is None:
        print(f"Error reading {filename}")
        continue

    h, w = image.shape[:2]
    resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(mask_path, resized_mask)
    print(f"Fixed {filename}")

print("Done")