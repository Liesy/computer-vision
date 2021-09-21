from cv2 import *
import numpy as np


def background_subtra(val):
    processed = np.sqrt(np.power(img - img_BG, 2))
    processed[processed >= val] = 255
    processed[processed < val] = 0
    imshow('processed', processed.astype(np.uint8))


img = imread('/home/liyang/study/cv/bgs/02.jpg', WINDOW_AUTOSIZE)
img_BG = imread('/home/liyang/study/cv/bgs/02_bg.jpg', WINDOW_AUTOSIZE)

if img.size != img_BG.size:
    print('img not match')
    exit(1)

namedWindow('original')
moveWindow('original', 200, 200)
imshow('original', img)

namedWindow('background')
moveWindow('background', 800, 200)
imshow('background', img_BG)

namedWindow('processed')
moveWindow('processed', 1400, 200)

createTrackbar('control', 'processed', 100, 255, background_subtra)

while True:
    k = waitKey(0)
    if k == 27:
        break

destroyAllWindows()
