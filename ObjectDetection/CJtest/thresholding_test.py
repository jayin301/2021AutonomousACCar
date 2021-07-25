import cv2 as cv
import numpy as np

#development environment
# OS:Windows 10
# Language: python 3.9.1
# Lib: opencv 4.5.3
#      numpy 1.19.3
#study by following the video url=https://www.youtube.com/watch?v=eLTLtUVuuy4
#study by following the document url=https://docs.opencv.org/3.4.1/d6/d00/tutorial_py_root.html

fname1 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul1.jpg'
rname11 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul11.jpg'
rname12 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul12.jpg'
fname2 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul2.jpg'
rname21 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul21.jpg'
rname22 = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Photos\\seoul22.jpg'

img1 = cv.imread(fname1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(fname2, cv.IMREAD_GRAYSCALE)

gau1 = cv.GaussianBlur(img1, (3,3), cv.BORDER_DEFAULT)
gau2 = cv.GaussianBlur(img2, (3,3), cv.BORDER_DEFAULT)

ret11, th11 =cv.threshold(gau1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
ret12, th12 =cv.threshold(gau2, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# gau11 = cv.GaussianBlur(th11, (3,3), cv.BORDER_DEFAULT)
# gau12 = cv.GaussianBlur(th12, (3,3), cv.BORDER_DEFAULT)
# gau21 = cv.GaussianBlur(th21, (3,3), cv.BORDER_DEFAULT)
# gau22 = cv.GaussianBlur(th22, (3,3), cv.BORDER_DEFAULT)

# bil11 = cv.bilateralFilter(th11, 9, 75, 75)
# bil12 = cv.bilateralFilter(th12, 9, 75, 75)
# bil21 = cv.bilateralFilter(th21, 9, 75, 75)
# bil22 = cv.bilateralFilter(th22, 9, 75, 75)

cv.imwrite(rname11, th11)
cv.imwrite(rname12, th12)
