import cv2
import numpy as np

img = cv2.imread("result.png", 0)

print("Unique values:", np.unique(img))