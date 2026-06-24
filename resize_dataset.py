import os
import cv2

img_dir = "data/imgs"
mask_dir = "data/masks"

size = (256, 256)

for filename in os.listdir(img_dir):

    img_path = os.path.join(img_dir, filename)

    if not filename.lower().endswith(".jpg"):
        continue

    mask_name = filename.replace(".jpg", ".png")
    mask_path = os.path.join(mask_dir, mask_name)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    if img is None or mask is None:
        print("Skipping", filename)
        continue

    img = cv2.resize(img, size)
    mask = cv2.resize(mask, size)

    cv2.imwrite(img_path, img)
    cv2.imwrite(mask_path, mask)

    print("Resized", filename)

print("Dataset resizing complete.")