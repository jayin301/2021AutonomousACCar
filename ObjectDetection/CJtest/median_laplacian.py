import cv2 as cv
import numpy as np
import random
import math

#development environment
# OS:Windows 10
# Language: python 3.9.1
# Lib: opencv 4.5.3
#      numpy 1.19.3
#study by following the video url=https://www.youtube.com/watch?v=eLTLtUVuuy4
#study by following the document url=https://docs.opencv.org/3.4.1/d6/d00/tutorial_py_root.html
#study by following the website url=https://www.programmersought.com/article/87395044475/
#study by following the website url=https://learnopencv.com/edge-detection-using-opencv/

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    #임시로 roi 설정
    polygons = np.array([[[width/7, height], [width*6/7, height], [width*5/7, height*2/3], [width*2/7, height*2/3]]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygons, 255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_image

fname = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Videos\\Seoul.mp4'
capture = cv.VideoCapture(fname)

#video should manage with frame-by-frame method
while True:
    isTrue, frame = capture.read()
    #image preprocessing : grayscale(for faster calculation), gaussian blur(for reducing noise)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray,3)
    #edge detection: lapalcian
    lap = cv.Laplacian(blur, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))


    #line detection
    #Method 1: Hough transform
    lane_image = np.copy(frame)
    cropped_image = region_of_interest(lap)
    lines = cv.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), 150, 10)
    # averaged_lines = average_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, lines)
    result = cv.addWeighted(lane_image, 0.7, line_image, 1, 1)

    cv.imshow('result', result)

    #Method 2: RANSAC algorithm

    #stop trigger
    if cv.waitKey(20)&0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()