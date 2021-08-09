import cv2 as cv
import numpy as np

#development environment
# OS:Windows 10
# Language: python 3.9.1c
# Lib: opencv 4.5.3
#      numpy 1.19.3

#demo버전임. 아직 코드 구현이 미완성임.
#manage 부분과 연결하려면 아마 객체지향으로 바꾸는 게 좋을 것 같긴 한데... 일단 유념해두기

def roi_warp(image):
    height = image.shape[0]
    width = image.shape[1]
    #관심영역과 동일하게 영상기반으로 임의로 설정
    befPoint = np.array([[0, height], [width, height], [width*4/5, height*0.6], [width/5, height*0.6]], dtype=np.float32)
    aftPoint = np.array([[[0, height], [width, height], [width, 0], [0, 0]]], dtype=np.float32)
    matrix = cv.getPerspectiveTransform(befPoint, aftPoint)
    warp = cv.warpPerspective(image, matrix, (width, height))
    return warp


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

fname = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Videos\\line.mp4'
capture = cv.VideoCapture(fname)

#video should manage with frame-by-frame method
while True:
    isTrue, frame = capture.read()
    aftframe = roi_warp(frame)
    #image preprocessing : resizing(640*480, for faster calculation), grayscale(for faster calculation), gaussian blur(for reducing noise)
    dst = cv.resize(aftframe, dsize = (640, 480), interpolation = cv.INTER_AREA)
    rot = cv.rotate(dst, cv.ROTATE_90_CLOCKWISE)
    gray = cv.cvtColor(rot, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)

    #차선 contour 얻어내는 방법
    #Method1(실선 얻어내기)
    # Processing Method
    canny = cv.Canny(gray, 100, 150)
    _, th = cv.threshold(canny, 100, 255, cv.THRESH_BINARY)
    #Hough Transform
    lane_image = np.copy(rot)
    lines = cv.HoughLines(th, 2, np.pi/180, 230)
    line_image = display_hough_lines(lane_image, lines)
    result = cv.addWeighted(lane_image, 0.7, line_image, 1, 1)

    #Method2(실선 얻어내기)
    # Processing Method
    _, thresh = cv.threshold(blur, 190, 255, cv.THRESH_BINARY)
    flip_thresh = cv.flip(thresh, 0)
    flipped_thresh = cv.flip(flip_thresh, 1)
    coordx = np.array([])
    coordy = np.array([])
    for x in range(480):
        for y in range(640):
            if flipped_thresh[y][x] != 0:
                coordx = np.append(coordx, np.array([x]))
                coordy = np.append(coordy, np.array([y]))
    # Polyfit
    # fit = np.polyfit(coordx,coordy,4)
    # a, b, c, d, e = fit
    # print(a,"x^4+",b,"x^3+",c,"x^2+",d,"x+",e)
    fit = np.polyfit(coordx, coordy, 1)
    a, b = fit
    print(a,"x+",b)

    #TFR+Labeling 이용하기(점선 얻어내기)
    contours, _ = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        area = cv.contourArea(approx)

        if len(approx) >= 3 and len(approx) <= 5:
            if area > 300:
                cv.drawContours(result, [approx], 0, (0, 255, 0), 5)

    #Labeling 위해서 contour 얻어내기
    # _, labels, stats, centroids = cv.connectedComponentsWithStats(th)
    # print(stats)
    # cv.imshow('er', th)

    cv.imshow('result',result)

    # stop trigger
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

