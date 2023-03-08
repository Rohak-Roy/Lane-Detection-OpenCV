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
pixelTolerance = tolerance * image.shape[1]

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
def getListOfDuplicateLinesIdx(lanes, pixelTolerance):
    duplicates = []
    result = []
    for idx in range(1, len(lanes)):
        prev = lanes[idx - 1][2]
        curr = lanes[idx][2]
        if abs(curr - prev) <= pixelTolerance:
            duplicates.append(idx - 1)
        else:
            duplicates.append(idx - 1)
            result.append(duplicates)
            duplicates = []
    
    lastItem = lanes[-1][2]
    secondLastItem = lanes[-2][2]
    if abs(lastItem - secondLastItem) <= pixelTolerance:
        duplicates.append(len(lanes) - 1)
        result.append(duplicates)
    else:
        result.append([len(lanes) - 1])

    return result

#TODO: Average the list of duplicate lanes amd return a list of unique lanes
def getUniqueLines(listOfLines, listOfDuplicateLinesIdx):
    uniqueLines = []
    for duplicateLinesIdx in listOfDuplicateLinesIdx:
        lines = []
        x1, y1, x2 ,y2 = 0, 0, 0, 0
        for lineIdx in duplicateLinesIdx:
            lines.append(listOfLines[lineIdx])
        for line in lines:
            x1 += line[0]
            y1 += line[1]
            x2 += line[2]
            y2 += line[3]
        avg_x1 = x1 // len(lines)
        avg_y1 = y1 // len(lines)
        avg_x2 = x2 // len(lines)
        avg_y2 = y2 // len(lines)
        uniqueLines.append([avg_x1, avg_y1, avg_x2, avg_y2])
    return uniqueLines

img_copy = image.copy()
grayImg = m.gray(img_copy)
gaussImg = m.gauss(grayImg)
cannyImg = m.canny(gaussImg)
maskedImg = m.getRegion(cannyImg)
lines = m.getLines(maskedImg)
slopes_and_y_intercepts = getSlopeAndY_Intercept(lines)
lanes = makeLines(image, slopes_and_y_intercepts)
listOfDuplicateLanesIdx = getListOfDuplicateLinesIdx(lanes, pixelTolerance)
uniqueLines = getUniqueLines(lanes, listOfDuplicateLanesIdx)

print(f'\n Number of lines originally detected = {len(lines)}')
print(f'\n Slopes and Y-Intercepts are = \n {slopes_and_y_intercepts}')
print(f'\n Co-ordinates of all proposed lanes are = \n {lanes}')
print(f'\n Indexes of duplicate lines are grouped together as = \n {listOfDuplicateLanesIdx}')
print(f'\n Final and unique lane co-ordinates  are = \n {uniqueLines}')

m.displayLinesOnImage(image, lines)

img_before_filtering = m.makeLinesOnBlackCanvas(image, lanes, (0, 180, 0))
m.showImage(img_before_filtering, 'Reconstructing All Lines Before Filtering on Image')     

uniqueLinesOnBlackCanvas = m.makeLinesOnBlackCanvas(image, uniqueLines)
m.showImage(uniqueLinesOnBlackCanvas, 'Unique Lines On Black Canvas')

output_image = cv2.addWeighted(image, 0.8, uniqueLinesOnBlackCanvas, 1, 1)
m.showImage(output_image, 'Final Image with Lane Detection')