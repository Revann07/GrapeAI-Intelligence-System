import os
import cv2

img_dir = "data/imgs"
mask_dir = "data/masks"

TARGET_SIZE = (512, 512)

for filename in os.listdir(img_dir):

    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(img_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    if image is None or mask is None:
        continue

    image = cv2.resize(image, TARGET_SIZE)
    mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(img_path, image)
    cv2.imwrite(mask_path, mask)

    print(f"Resized {filename}")

print("All images & masks resized.")