import cv2 as cv
import numpy as np


#development environment
# OS:Windows 10
# Language: python 3.9.1c
# Lib: opencv 4.5.3
#      numpy 1.19.3

def roi_warp(image):
    height = image.shape[0]
    width = image.shape[1]
    #관심영역과 동일하게 영상기반으로 임의로 설정
    befPoint = np.array([[width/4, height*3/4], [width*3/4, height*3/4], [width*3/5, height*0.56], [width*2/5, height*0.56]], dtype=np.float32)
    aftPoint = np.array([[[0, height], [width, height], [width, 0], [0, 0]]], dtype=np.float32)
    matrix = cv.getPerspectiveTransform(befPoint, aftPoint)
    warp = cv.warpPerspective(image, matrix, (width, height))
    return warp

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    #영상기반으로 임의지정한 관심영역
    polygons = np.array([[[width/4, height*3/4], [width*3/4, height*3/4], [width*3/5, height*0.5], [width*2/5, height*0.5]]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygons, 255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1, y1), (x2, y2), (255,0,0), 5)
    return line_image

def display_hough_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv.line(line_image, (x1,y1), (x2, y2), (0,0,255), 5)
    return line_image

def TFR(image, greyimage):
    _, thresh = cv.threshold(greyimage, 100 ,255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01* cv.arcLength(contour, True), True)
        area = cv.contourArea(approx)

        if len(approx) >=3 and len(approx) <=5 :
            if area > 200:
                cv.drawContours(image, [approx], 0, (0, 255, 0), 5)

    return image

fname = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Videos\\1.avi'
capture = cv.VideoCapture(fname)

#video should manage with frame-by-frame method
while True:
    isTrue, frame = capture.read()
    #image preprocessing : resizing(640*480, for faster calculation), grayscale(for faster calculation), gaussian blur(for reducing noise)
    dst = cv.resize(frame, dsize = (640, 480), interpolation = cv.INTER_AREA)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
    #edge detection : canny edge detector
    canny = cv.Canny(gray, 100, 150)

    #line detection
    #Method 1: Simple Probabilistic Hough transform
    lane_image = np.copy(dst)
    cropped_image = region_of_interest(canny)
    lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), 70, 10)
    line_image = display_lines(lane_image, lines)
    result = cv.addWeighted(lane_image, 0.7, line_image, 1, 1)

    #Method 2: Top view Probabilistic Hough transform
    warped_image = roi_warp(canny)
    warp_lane_image = np.copy(dst)
    warped_lane_image = roi_warp(warp_lane_image)
    warp_lines = cv.HoughLinesP(warped_image, 2, np.pi/180, 650, np.array([]), 70, 100)
    warp_line_image = display_lines(warped_lane_image, warp_lines)
    warp_result = cv.addWeighted(warped_lane_image, 0.7, warp_line_image, 1, 1)

    #Method 3: Hough transform
    hough_image = np.copy(dst)
    hough_cropped_image = region_of_interest(canny)
    hough_lines = cv.HoughLines(hough_cropped_image, 2, np.pi/180, 130)
    hough_line_image = display_hough_lines(hough_image, hough_lines)
    hough_result = cv.addWeighted(hough_image, 0.7, hough_line_image, 1, 1)

    #Method 4: Top view Hough transform
    warped_hough_image = roi_warp(canny)
    warp_hough_lane_image = np.copy(dst)
    warped_hough_lane_image = roi_warp(warp_hough_lane_image)
    warp_hough_lines = cv.HoughLines(warped_hough_image, 2, np.pi/180, 800)
    warp_hough_line_image = display_hough_lines(warped_hough_lane_image, warp_hough_lines)
    warp_hough_result = cv.addWeighted(warped_hough_lane_image, 0.7, warp_hough_line_image, 1, 1)

    #Method 5: TFR(Top view Finding Rectangle) algorithm
    tfr_image = roi_warp(gray)
    bef_tfr_lane_image = np.copy(dst)
    aft_tfr_lane_image = roi_warp(bef_tfr_lane_image)
    tfr_result = TFR(aft_tfr_lane_image, tfr_image)

    # lane detection result
    cv.imshow('houghP', result)
    cv.imshow('warp_houghP', warp_result)
    cv.imshow('hough', hough_result)
    cv.imshow('warp_hough', warp_hough_result)
    cv.imshow('tfr', tfr_result)
    cv.imshow('original', dst)

    #stop trigger
    if cv.waitKey(20)&0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()