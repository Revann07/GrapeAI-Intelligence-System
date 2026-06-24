import cv2
import numpy as np
import subprocess
import sys

if len(sys.argv) < 2:
    print("Usage: python demo.py image_path")
    exit()

image_path = sys.argv[1]

print("\nRunning canopy segmentation...\n")

subprocess.run([
    "python",
    "predict.py",
    "-m",
    "checkpoints/checkpoint_epoch30.pth",
    "-i",
    image_path,
    "-o",
    "result_demo.png"
])

mask = cv2.imread("result_demo.png", 0)
img = cv2.imread(image_path)

leaf_pixels = np.sum(mask > 0)
total_pixels = mask.shape[0] * mask.shape[1]
coverage = (leaf_pixels / total_pixels) * 100

print(f"\nCanopy Coverage: {coverage:.2f} %")

mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
combined = np.hstack((img, mask_color))

cv2.imshow("Input Image  |  Predicted Leaf Mask", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()