from cv2 import *
import numpy as np


def controller(sigmoid):
    processed = np.multiply(img, 1 / (np.exp(-np.multiply(np.multiply(img - 127.5, 1 / 255), 0.1 * sigmoid)) + 1)) + 0.5
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    imshow('exp1-1_processed', processed)


img = imread('/home/liyang/图片/einstein.jpeg')
print(type(img), img.shape, img.dtype)

namedWindow('exp1-1_original', WINDOW_AUTOSIZE)
moveWindow('exp1-1_original', 200, 200)
imshow('exp1-1_original', img)

namedWindow('exp1-1_processed', WINDOW_AUTOSIZE)
moveWindow('exp1-1_processed', 800, 200)
imshow('exp1-1_processed', img)

createTrackbar('contrast control', 'exp1-1_processed', 0, 100, controller)

while True:
    k = waitKey(0)
    if k == 27:
        break

destroyAllWindows()
