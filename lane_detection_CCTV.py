import cv2
import os
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy as np
import re
from tqdm import tqdm
import methods as m

image = cv2.imread('2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tolerance = 0.1
pixelTolerance = 0.1 * image.shape[1]

def getSlopeAndY_Intercept(lines):
    list = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_intercept = parameters[1]
            tuple = (slope, y_intercept)
            if abs(slope) < 1:
                continue
            list.append(tuple)
    return np.array(list)

def makeLines(image, tuples):
    list = []
    for tuple in tuples:
        slope, y_intercept = tuple
        y1 = image.shape[0]
        y2 = 0
        x1 = int((y1 - y_intercept) // slope)
        x2 = int((y2 - y_intercept) // slope)
        list.append([x1, y1, x2, y2])
    list.sort(key=lambda f: f[2])
    return np.array(list)

#TODO: Return a list of duplicate lanes
def checkForDuplicateLines(lanes, tolerance):
    pass

#TODO: Average the list of duplicate lanes amd return a list of unique lanes

#TODO: Write a function to get the number of lanes

img_copy = image.copy()
grayImg = m.gray(img_copy)
gaussImg = m.gauss(grayImg)
cannyImg = m.canny(gaussImg)
maskedImg = m.getRegion(cannyImg)
lines = m.getLines(maskedImg)
slopes_and_y_intercepts = getSlopeAndY_Intercept(lines)
lanes = makeLines(image, slopes_and_y_intercepts)
print(f'Number of lines originally detected = {len(lines)}')
print(f'Slopes and Y-Intercepts are = {slopes_and_y_intercepts}')
print(f'Lanes are = {lanes}')

for lane in lanes:
    img = m.makeLinesOnBlackCanvas(image, [lane])
    plt.imshow(img)
    plt.show()

# img = m.makeLinesOnBlackCanvas(image, lanes)
# plt.imshow(img)
# plt.show()             