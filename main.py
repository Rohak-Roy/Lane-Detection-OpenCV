import os

import cv2
from matplotlib import pyplot as plt

path = 'C:\Rohak\OpenCV - Toshiba\sample_image.jpg'
img = cv2.imread(path)

plt.figure(figsize=(10, 10))
plt.imshow(img[:, :, 0], cmap='gray')
plt.show()