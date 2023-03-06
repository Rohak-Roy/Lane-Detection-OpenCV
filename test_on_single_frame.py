import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import re
from tqdm import tqdm

path = 'frames'
col_frames = os.listdir(path)
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

col_images = []
for frame in tqdm(col_frames):
    img = cv2.imread(path + '/' + frame)
    col_images.append(img)

idx = 245
original_img_colored = col_images[idx]
example_img = col_images[idx][:, :, 0]
figure = plt.figure(figsize=(10, 10))
rows, cols = 3, 2

stencil = np.zeros_like(example_img)
polygon = np.array([[50, 270], [220, 160], [360, 160], [480, 270]])
cv2.fillConvexPoly(stencil, polygon, 1)

masked_img = cv2.bitwise_and(example_img, example_img, mask = stencil)

_, thresh = cv2.threshold(masked_img, 130, 145, cv2.THRESH_BINARY)

lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 20, maxLineGap = 200)
original_img_colored_copy = original_img_colored.copy()
print("NUMBER OF LINES = ", len(lines))

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(original_img_colored_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)

figure.add_subplot(rows, cols, 1)
plt.imshow(original_img_colored)
plt.title("Original image")

figure.add_subplot(rows, cols, 2)
plt.imshow(example_img, cmap = 'gray')
plt.title("Grayscaled original image")

figure.add_subplot(rows, cols, 3)
plt.imshow(stencil, cmap = 'gray')
plt.title("Created Mask")

figure.add_subplot(rows, cols, 4)
plt.imshow(masked_img, cmap = 'gray')
plt.title("Bitwise AND mask with grayscaled image")

figure.add_subplot(rows, cols, 5)
plt.imshow(thresh, cmap = 'gray')
plt.title("Thresholded image")

figure.add_subplot(rows, cols, 6)
plt.imshow(original_img_colored_copy)
plt.title("Used Hough Transform to detect lanes")
plt.show()