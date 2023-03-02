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
if 'detected' in parent_dir_files:
    detection_complete = True

images = []
print("READING ALL IMAGES INTO A LIST:")
for frame in tqdm(frames_list):
    img = cv2.imread(path_frames + '/' + frame)
    images.append(img)

height, width, depth = images[0].shape
size = (width, height)

stencil = np.zeros_like(images[0][:, :, 0])
polygon = np.array([[50, 270], [220, 160], [360, 160], [480, 270]])
cv2.fillConvexPoly(stencil, polygon, 1)

if detection_complete == False:
    os.mkdir('detected')
    current_dir = os.chdir('detected')
    print(f"PERFORMING LANE DETECTION FOR ALL {len(images)} IMAGES:")
    for image in tqdm(images):
        masked_image = cv2.bitwise_and(image[:, :, 0], image[:, :, 0], mask = stencil)
        _, thresholded_image = cv2.threshold(masked_image, 130, 145, cv2.THRESH_BINARY)
        lines = cv2.HoughLinesP(thresholded_image, 1, np.pi/180, 30, maxLineGap = 200)
        image_copy = image.copy()

        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.imwrite(str(count) + '.png', image_copy)

        except TypeError:
            print('ERROR OCCURRED WHILE SAVING IMAGE ' + str(count) + '.png')
            cv2.imwrite(str(count) + '.png', image)

        count += 1

current_dir = os.chdir(parent_dir)

pathIn = 'detected/'
pathOut = 'roads.mp4'
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