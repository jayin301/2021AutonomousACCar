import cv2 as cv
import numpy as np

#development environment
# OS:Windows 10
# Language: python 3.9.1
# Lib: opencv 4.5.3
#      numpy 1.19.3

fname1 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul1.jpg'
rname11 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul11.jpg'
rname12 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul12.jpg'
fname2 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul2.jpg'
rname21 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul21.jpg'
rname22 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul22.jpg'

# 이하 blur test
# gray1 = cv.imread(fname1, cv.IMREAD_GRAYSCALE)
# gray2 = cv.imread(fname2, cv.IMREAD_GRAYSCALE)
# avg1 = cv.blur(gray1, (3,3))
# avg2 = cv.blur(gray2, (3,3))
# med1 = cv.medianBlur(gray1, 3)
# med2 = cv.medianBlur(gray2, 3)
# gau1 = cv.GaussianBlur(gray1, (3,3), cv.BORDER_DEFAULT)
# gau2 = cv.GaussianBlur(gray2, (3,3), cv.BORDER_DEFAULT)
# bil1 = cv.bilateralFilter(gray1, 9, 75, 75)
# bil2 = cv.bilateralFilter(gray2, 9, 75, 75)

# canny1 = cv.Canny(bil1, 100, 150)
# lap1 = cv.Laplacian(bil1, cv.CV_64F)
# lap1 = np.uint8(np.absolute(lap1))
#
# canny2 = cv.Canny(bil2, 100, 150)
# lap2 = cv.Laplacian(bil2, cv.CV_64F)
# lap2 = np.uint8(np.absolute(lap2))


cv.imwrite(rname11, canny1)
cv.imwrite(rname12, lap1)
cv.imwrite(rname21, canny2)
cv.imwrite(rname22, lap2)