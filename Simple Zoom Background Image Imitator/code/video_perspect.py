import string
import cv2 
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ffmpeg', type=str, nargs='?', default='0')
args = parser.parse_args()

video_path = "../data/test.mp4"
ffmpeg_video = False
if (args.ffmpeg.strip().lower() == "true"):
    ffmpeg_video = True
    video_path = "../data/test_2.mp4"

img1 = cv2.imread("../data/1.jpg") # Reading image
# img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
vid = cv2.VideoCapture(video_path) # Reading video
# arch = cv2.cvtColor(arch,cv2.COLOR_BGR2RGB)
print(video_path)
row, col, ch= img1.shape

pts1 = np.array([[1200, 900], [1200, 0], [0, 0], [0, 900]]).astype(np.float32)
pts2 = np.array([[2998, 2050], [2954, 726], [1512, 169], [1493, 2221]]).astype(np.float32)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
frame_rate = 8
prev = 0
while True:
    ret, frame = vid.read()
    if not ret : break

    time_elapsed = time.time() - prev

    if  ffmpeg_video  or (time_elapsed > 1./frame_rate):
        prev = time.time()

        x,y = frame.shape[0], frame.shape[1]
        pts1 = np.array([ [y,x], [y,0], [0,0], [0,x] ]).astype(np.float32)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (col, row))

        mask = np.zeros(img1.shape, dtype= np.uint8)

        mask_frame = cv2.fillConvexPoly(mask, np.array([pts2], dtype=np.int32), (255,255,255))
        mask_frame = cv2.bitwise_not(mask_frame)
        
        hollow_frame = cv2.bitwise_and(img1, mask_frame)
        final_frame = cv2.bitwise_or(hollow_frame, result)

        cv2.imshow('frame', final_frame)

    if cv2.waitKey(1) == ord('q'): break


vid.release()
# Destroy all the windows
cv2.destroyAllWindows()