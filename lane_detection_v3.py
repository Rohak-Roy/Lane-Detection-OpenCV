import cv2
import os
from os.path import isfile, join
from matplotlib import pyplot as plt
import numpy as np
import re
from tqdm import tqdm

detection_complete = False
count = 0
path_frames = 'frames'
frames_list = os.listdir(path_frames)
frames_list.sort(key=lambda f: int(re.sub('\D', '', f)))

parent_dir = os.getcwd()
current_dir = parent_dir
parent_dir_files = os.listdir(parent_dir)
if 'detected_v3' in parent_dir_files:
    detection_complete = True

images = []
print("READING ALL IMAGES INTO A LIST:")
for frame in tqdm(frames_list):
    img = cv2.imread(path_frames + '/' + frame)
    images.append(img)

height, width, depth = images[0].shape
size = (width, height)

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

if detection_complete == False:
    os.mkdir('detected_v3')
    current_dir = os.chdir('detected_v3')
    print(f"PERFORMING LANE DETECTION FOR ALL {len(images)} IMAGES:")
    for image in tqdm(images):
        img_copy = image.copy()
        output_image = image.copy()

        img_copy = gray(img_copy)
        img_copy = gauss(img_copy)
        img_copy = canny(img_copy)
        img_copy = region(img_copy)
        lines = getLines(img_copy)
        averaged_lines = average(lines)
        black_lines = displayLines(output_image, averaged_lines)
        lanes = cv2.addWeighted(output_image, 0.8, black_lines, 1, 1)

        try:
            cv2.imwrite(str(count) + '.png', lanes)
        except TypeError:
            print('ERROR OCCURRED WHILE SAVING IMAGE ' + str(count) + '.png')
            cv2.imwrite(str(count) + '.png', image)
        count += 1

current_dir = os.chdir(parent_dir)

pathIn = 'detected_v3/'
pathOut = 'roads_v3.mp4'
fps = 60.0

detected_files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
detected_files.sort(key=lambda f: int(re.sub('\D', '', f)))

detected_frames_list = []
for idx in tqdm(range(len(detected_files))):
    filename = pathIn + detected_files[idx]
    img = cv2.imread(filename)
    detected_frames_list.append(img)

out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for idx in range(len(detected_frames_list)):
    out.write(detected_frames_list[idx])

out.release()