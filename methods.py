import cv2
from matplotlib import pyplot as plt
import numpy as np

def gray(image):
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def canny(image):
    edges = cv2.Canny(image, 140, 150)
    return edges

def binaryThreshold(image):
    _, thresholded_image = cv2.threshold(image, 1, 200, cv2.THRESH_BINARY)
    return thresholded_image

def getRegion(image):
    mask_height, mask_width = 500, 720
    polygon = np.array([(350, 0), (640, 0), (mask_width, mask_height), (70, mask_height)])
    mask = np.zeros_like(image)
    mask = cv2.fillConvexPoly(mask, polygon, 1)
    mask = cv2.bitwise_and(image, mask)
    return mask

def getLines(image):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=30, maxLineGap=200)
    return lines

def average(lines, image):
    left = []
    right = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            y_intercept = parameters[1]
            if slope < 0:
                left.append((slope, y_intercept))
            else:
                right.append((slope, y_intercept))
    
    if len(left) == 0:
        left_avg = np.array([])
    else:
        left_avg = np.average(left, axis=0)
    if len(right) == 0:
        right_avg = np.array([])
    else:
        right_avg = np.average(right, axis=0)

    left_line = makePoints(image, left_avg)
    right_line = makePoints(image, right_avg)
    return np.array([left_line, right_line])

def makePoints(image, average):
    if len(average) == 0:
        return np.array([0, 0, 0, 0])
    
    slope, y_intercept = average
    y1 = image.shape[0]
    y2 = int(y1 * 2.85/5)
    x1 = int((y1 - y_intercept) // slope)  
    x2 = int((y2 - y_intercept) // slope)
    return np.array([x1, y1, x2, y2])

def makeLinesOnBlackCanvas(image, lines, colour=(0, 255, 0)):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), colour, 9)
    return lines_image

#For debugging purposes.
def displayLinesOnImage(image, lines):
    if lines is None:
        return
    
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

def showImage(image, title, grayscale=True):
    plt.figure(figsize=(7, 7))
    if grayscale == True:
        plt.imshow(image, cmap = 'gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def getSlope(line):
    x1, y1, x2, y2 = line
    parameters = np.polyfit((x1, x2), (y1, y2), 1)
    slope = parameters[0]
    return slope

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
