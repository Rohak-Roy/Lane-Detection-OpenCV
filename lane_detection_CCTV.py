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

def get_slope_and_y_intercept(lines):
    list = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_intercept = parameters[1]
            pair = (slope, y_intercept)
            list.append(pair)
    return np.array(list)

#TODO: USE LINE EQUATION TO DRAW LINES

img_copy = image.copy()
grayImg = m.gray(img_copy)
gaussImg = m.gauss(grayImg)
cannyImg = m.canny(gaussImg)
maskedImg = m.region(cannyImg)
lines = m.getLines(maskedImg)