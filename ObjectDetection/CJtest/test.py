import cv2 as cv
import numpy as np

#development environment
# OS:Windows 10
# Language: python 3.9.1
# Lib: opencv 4.5.3
#      numpy 1.19.3
#study by following the video url=https://www.youtube.com/watch?v=oXlwWbU8l2o

fname = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Videos\\Seoul.mp4'
capture = cv.VideoCapture(fname)
#video should manage with frame-by-frame method
while True:
    isTrue, frame = capture.read()
    #image preprocessing prac
    #idea: maybe we can use thresholding instead of grayscale convertion
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
    #edge detection
    #1st: canny edge detector
    canny = cv.Canny(gray, 100, 150)
    #2nd: lapalcian
    lap = cv.Laplacian(gray, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))
    #3rd: sobel
    sobelx=cv.Sobel(blur, cv.CV_64FC1,1,0)
    sobely=cv.Sobel(blur, cv.CV_64FC1,0,1)
    combined_sobel = cv.bitwise_or(sobelx, sobely)
    #window open
    # cv.imshow('original',frame)
    # cv.imshow('grayscale', gray)
    # cv.imshow('grayscale&blur',blur)
    cv.imshow('Canny', canny)
    cv.imshow('Laplacian',lap)
    cv.imshow('Sobel', combined_sobel)
    #stop trigger
    if cv.waitKey(20)&0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
