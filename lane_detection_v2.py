import cv2
import os
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy as np
import re
from tqdm import tqdm

frames_list = os.listdir('frames')
frames_list.sort(key = lambda f: int(re.sub('\D', '', f)))

image_path = 'frames/' + frames_list[0]
image = cv2.imread(image_path)

def gray(image):
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def canny(image):
    edges = cv2.Canny(image, 50, 150)
    return edges

def region(image):
    height, width = image.shape
    polygon = np.array([[50, 270], [220, 160], [360, 160], [480, 270]])
    mask = np.zeros_like(image)
    mask = cv2.fillConvexPoly(mask, polygon, 1)
    mask = cv2.bitwise_and(image, mask)
    return mask

def getLines(image):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=30, maxLineGap=200)
    return lines

def average(lines):
    left = []
    right = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    left_line = makePoints(image, left_avg)
    right_line = makePoints(image, right_avg)
    return np.array([left_line, right_line])

def makePoints(image, average):
    slope, y_intercept = average
    y1 = image.shape[0]
    y2 = int(y1 * 3/5)
    x1 = int((y1 - y_intercept) // slope)  
    x2 = int((y2 - y_intercept) // slope)
    return np.array([x1, y1, x2, y2])

def displayLines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 180, 0), 9)
    return lines_image

#For debugging purposes.
def displayLineCoordinates(image, lines):
    fig = plt.figure(figsize=(10, 10))
    rows, cols = 1, len(lines)
    for idx, line in enumerate(lines):
        debugImg = image.copy()
        x1, y1, x2, y2 = line[0] if len(lines.shape) > 2 else line 
        #x1, y1, x2, y2 = line
        cv2.line(debugImg, (x1, y1), (x2, y2), (0, 0, 255), 9)
        fig.add_subplot(rows, cols, idx + 1)
        plt.title(f"Line: {idx + 1}.")
        plt.imshow(debugImg)
    plt.show()

img_copy = image.copy()
img_copy = gray(img_copy)
img_copy = gauss(img_copy)
img_copy = canny(img_copy)
img_copy = region(img_copy)
lines = getLines(img_copy)
averaged_lines = average(lines)
displayLineCoordinates(image, lines)
black_lines = displayLines(image, averaged_lines)
lanes = cv2.addWeighted(image, 0.8, black_lines, 1, 1)
plt.figure(figsize=(10, 10))
plt.imshow(lanes)
plt.show()