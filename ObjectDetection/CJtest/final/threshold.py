import numpy as np
import cv2

def sobel_xy(img, orient='x', thresh=(20, 100)):

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0.7, 1.3)):

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255

    return binary_output.astype(np.uint8)

def ch_thresh(ch, thresh=(80, 255)):
    binary = np.zeros_like(ch)
    binary[(ch > thresh[0]) & (ch <= thresh[1])] = 255
    return binary

def gradient_combine(img, th_x, th_y, th_mag, th_dir):
    rows, cols = img.shape[:2]
    R = img[200:rows - 110, 0:cols, 2]

    sobelx = sobel_xy(R, 'x', th_x)
    #cv2.imshow('sobel_x', sobelx)
    sobely = sobel_xy(R, 'y', th_y)
    #cv2.imshow('sobel_y', sobely)
    mag_img = mag_thresh(R, 3, th_mag)
    #cv2.imshow('sobel_mag', mag_img)
    dir_img = dir_thresh(R, 15, th_dir)
    #cv2.imshow('result5', dir_img)

    gradient_comb = np.zeros_like(dir_img).astype(np.uint8)
    gradient_comb[((sobelx > 1) & (mag_img > 1) & (dir_img > 1)) | ((sobelx > 1) & (sobely > 1))] = 255

    return gradient_comb

def hls_combine(img, th_h, th_l, th_s):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    rows, cols = img.shape[:2]

    H = hls[200:rows - 110, 0:cols, 0]
    L = hls[200:rows - 110, 0:cols, 1]
    S = hls[200:rows - 110, 0:cols, 2]

    h_img = ch_thresh(H, th_h)
    # cv2.imshow('HLS (H) threshold', h_img)
    l_img = ch_thresh(L, th_l)
    # cv2.imshow('HLS (L) threshold', l_img)
    s_img = ch_thresh(S, th_s)
    # cv2.imshow('HLS (S) threshold', s_img)

    hls_comb = np.zeros_like(s_img).astype(np.uint8)
    hls_comb[((s_img > 1) & (l_img == 0)) | ((l_img == 0) & (h_img > 1) & (s_img > 1))] = 255
    return hls_comb

def comb_result(grad, hls):
    result = np.zeros_like(hls).astype(np.uint8)

    result[(grad > 1)] = 100
    result[(hls > 1)] = 255

    return result
