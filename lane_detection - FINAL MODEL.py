import cv2
import methods as m
import matplotlib.pyplot as plt

image = cv2.imread('2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tolerance = 0.1
pixelTolerance = tolerance * image.shape[1]

img_copy = image.copy()
print(img_copy.shape)
grayImg = m.gray(img_copy)
gaussImg = m.gauss(grayImg)
cannyImg = m.canny(gaussImg)
maskedImg = m.getRegion(cannyImg)
lines = m.getLines(maskedImg)
slopes_and_y_intercepts = m.getSlopeAndY_Intercept(lines)
lanes = m.makeLines(image, slopes_and_y_intercepts)
listOfDuplicateLanesIdx = m.getListOfDuplicateLinesIdx(lanes, pixelTolerance)
uniqueLines = m.getUniqueLines(lanes, listOfDuplicateLanesIdx)

print(f'\n Number of lines originally detected = {len(lines)}')
print(f'\n Slopes and Y-Intercepts are = \n {slopes_and_y_intercepts}')
print(f'\n Co-ordinates of all proposed lanes are = \n {lanes}')
print(f'\n Indexes of duplicate lines are grouped together as = \n {listOfDuplicateLanesIdx}')
print(f'\n Final and unique lane co-ordinates in the form of [(x1, y1), (x2, y2)] are = \n {uniqueLines}')

m.displayLinesOnImage(image, lines)

img_before_filtering = m.makeLinesOnBlackCanvas(image, lanes, (0, 180, 0))
m.showImage(img_before_filtering, 'Reconstructing All Lines Before Filtering on Image')     

uniqueLinesOnBlackCanvas = m.makeLinesOnBlackCanvas(image, uniqueLines)
m.showImage(uniqueLinesOnBlackCanvas, 'Unique Lines On Black Canvas')

output_image = cv2.addWeighted(image, 0.8, uniqueLinesOnBlackCanvas, 1, 1)
m.showImage(output_image, 'Final Image with Lane Detection')