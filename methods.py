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

def region(image):
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

def displayLines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 180, 0), 6)
    return lines_image

#For debugging purposes.
def displayLineCoordinates(image, lines):
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
    plt.figure(figsize=(10, 10))
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