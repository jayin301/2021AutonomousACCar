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

# def make_coordinates(image, line_parameters):
#     try:
#         slope, intercept = line_parameters
#     except TypeError:
#         slope, intercept = 0.001, 0
#     y1 = image.shape[0]
#     y2 = int(y1*(3/5))
#     x1 = int((y1 - intercept)/slope)
#     x2 = int((y2 - intercept)/slope)
#     return np.array([x1, y1, x2, y2])

# def average_slope_intercept(image, lines):
#     left_fit = []
#     right_fit = []
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line.reshape(4)
#             parameters = np.polyfit((x1,x2), (y1,y2), 1)
#             slope = parameters[0]
#             intercept = parameters[1]
#             if slope <0:
#                 left_fit.append((slope, intercept))
#             else:
#                 right_fit.append((slope, intercept))
#         left_fit_average = np.average(left_fit, 0)
#         right_fit_average = np.average(right_fit, 0)
#         left_line = make_coordinates(image, left_fit_average)
#         right_line = make_coordinates(image, right_fit_average)
#         return np.array([left_line, right_line])
#     else:
#         return lines

fname = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Videos\\Seoul.mp4'
capture = cv.VideoCapture(fname)

#video should manage with frame-by-frame method
while True:
    isTrue, frame = capture.read()
    #image preprocessing : grayscale(for faster calculation), gaussian blur(for reducing noise)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
    #edge detection : canny edge detector
    canny = cv.Canny(blur, 100, 150)

    #line detection
    #Method 1: Hough transform
    lane_image = np.copy(frame)
    cropped_image = region_of_interest(canny)
    lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), 70, 10)
    #averaged_lines = average_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, lines)
    result = cv.addWeighted(lane_image, 0.7, line_image, 1, 1)

    cv.imshow('result', result)

    #Method 2: RANSAC algorithm(다음주 구현하고 비교 테스트 진행)


    #stop trigger
    if cv.waitKey(20)&0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()