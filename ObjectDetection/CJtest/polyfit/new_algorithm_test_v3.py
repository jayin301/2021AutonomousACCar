import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pickle

#trial and error로 mask 형태는 바꿔주기
#functions for change perspective of camera
def image_warp_mtx(img):
    height = img.shape[0]
    width = img.shape[1]
    src = np.float32([[width/3, height], [width, height], [width * 4 / 5, height * 0.6], [width*0.45, height * 0.6]])
    dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
    return src, dst

def perspect(img,src,dst):
    height = img.shape[0]
    width = img.shape[1]
    M= cv2.getPerspectiveTransform(src,dst)
    Minv= cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img,M,(width, height),flags =  cv2.INTER_LINEAR)
    return warped, M, Minv

#thresholding
# def sobel(image, sx_thresh=(20, 100)):
#     bgr = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)
#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
#     abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
#     scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
#     return scaled_sobel

#startpoint찾기
def plothistogram(image):
    histogram = np.sum(image[int(image.shape[0]*0.95):, :], axis=0)
    #histogram check
    # height, width = image.shape
    # blankImage = np.zeros((height, width, 3), np.uint8)
    # for col in range(width):
    #     if histogram[col]>1000:
    #         cv2.line(blankImage, (col, 0), (col,int(histogram[col]*height/width)), (255, 255, 255), 1)
    # cv2.imshow('his',blankImage)
    midpoint = np.int(histogram.shape[0] / 2)
    leftbase = np.argmax(histogram[:midpoint])
    rightbase = np.argmax(histogram[midpoint:]) + midpoint

    if histogram[leftbase] <500:
        leftbase =0
    if histogram[rightbase] <500:
        rightbase =0
    #print(leftbase, rightbase)

    return leftbase, rightbase

#case 별로 line 찾기
def window_search(binary_warped, left_current, right_current):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    nwindows = 15
    window_height = np.int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()  # 선이 있는 부분의 인덱스만 저장
    nonzero_y = np.array(nonzero[0])  # 선이 있는 부분 y의 인덱스 값
    nonzero_x = np.array(nonzero[1])  # 선이 있는 부분 x의 인덱스 값
    margin = 150
    minpix = 50
    left_lane = []
    right_lane = []
    color = [0, 255, 0]
    thickness = 2

    if (left_current!=0) and (right_current!=0):
        for w in range(nwindows):
            win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
            win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
            win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
            win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
            win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
            win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
            good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
            left_lane.append(good_left)
            right_lane.append(good_right)

            if len(good_left) > minpix:
                left_current = np.int(np.mean(nonzero_x[good_left]))
            if len(good_right) > minpix:
                right_current = np.int(np.mean(nonzero_x[good_right]))

        left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침
        right_lane = np.concatenate(right_lane)

        leftx = nonzero_x[left_lane]
        lefty = nonzero_y[left_lane]
        rightx = nonzero_x[right_lane]
        righty = nonzero_y[right_lane]

        left_fit = np.polyfit(lefty, leftx, 1)
        right_fit = np.polyfit(righty, rightx, 1)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty + left_fit[1]
        right_fitx = right_fit[0] * ploty + right_fit[1]

        ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림
        rtx = np.trunc(right_fitx)

        out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
        out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

        ret = {'left_fitx' : ltx, 'right_fitx': rtx, 'ploty': ploty}

    elif left_current == 0:
        for w in range(nwindows):
            win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
            win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
            win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
            win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
            win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
            win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
            good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
                        nonzero_x < win_xleft_high)).nonzero()[0]
            good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                        nonzero_x < win_xright_high)).nonzero()[0]
            left_lane.append(good_left)
            right_lane.append(good_right)

            if len(good_right) > minpix:
                right_current = np.int(np.mean(nonzero_x[good_right]))

        right_lane = np.concatenate(right_lane)

        rightx = nonzero_x[right_lane]
        righty = nonzero_y[right_lane]

        right_fit = np.polyfit(righty, rightx, 1)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        right_fitx = right_fit[0] * ploty + right_fit[1]

        rtx = np.trunc(right_fitx)

        out_img[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

        plt.imshow(out_img)
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

        ret = {'left_fitx' : 0,'right_fitx': rtx, 'ploty': ploty}

    else:
        for w in range(nwindows):
            win_y_low = binary_warped.shape[0] - (w + 1) * window_height  # window 윗부분
            win_y_high = binary_warped.shape[0] - w * window_height  # window 아랫 부분
            win_xleft_low = left_current - margin  # 왼쪽 window 왼쪽 위
            win_xleft_high = left_current + margin  # 왼쪽 window 오른쪽 아래
            win_xright_low = right_current - margin  # 오른쪽 window 왼쪽 위
            win_xright_high = right_current + margin  # 오른쪽 window 오른쪽 아래

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, thickness)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, thickness)
            good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
                        nonzero_x < win_xleft_high)).nonzero()[0]
            good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                        nonzero_x < win_xright_high)).nonzero()[0]
            left_lane.append(good_left)

            if len(good_left) > minpix:
                left_current = np.int(np.mean(nonzero_x[good_left]))

        left_lane = np.concatenate(left_lane)  # np.concatenate() -> array를 1차원으로 합침

        leftx = nonzero_x[left_lane]
        lefty = nonzero_y[left_lane]

        left_fit = np.polyfit(lefty, leftx, 1)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty + left_fit[1]

        ltx = np.trunc(left_fitx)  # np.trunc() -> 소수점 부분을 버림

        out_img[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]

        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

        ret = {'left_fitx': ltx,'right_fitx': 0, 'ploty': ploty}

    return ret

#원래 영상에 라인 그려주기
# def draw_lane_lines(original_image, warped_image, Minv, draw_info):
#
#     left_fitx = draw_info['left_fitx']
#     right_fitx = draw_info['right_fitx']
#     ploty = draw_info['ploty']
#
#     if (left_fitx != 0) and (right_fitx!=0):
#         warp_zero = np.zeros_like(warped_image).astype(np.uint8)
#         color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#
#         pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
#         pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
#         pts = np.hstack((pts_left, pts_right))
#
#         mean_x = np.mean((left_fitx, right_fitx), axis=0)
#         pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
#
#         cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
#         cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))
#
#         newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
#         result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)
#
#     elif left_fitx==0:
#         warp_zero = np.zeros_like(warped_image).astype(np.uint8)
#         color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#
#         pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
#         pts = np.hstack((pts_right))
#
#         mean_x = np.mean((right_fitx), axis=0)
#         pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
#
#         cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
#         cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))
#
#         newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
#         result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)
#
#     else:
#         warp_zero = np.zeros_like(warped_image).astype(np.uint8)
#         color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#
#         pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
#         pts = np.hstack((pts_left))
#
#         mean_x = np.mean((left_fitx), axis=0)
#         pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
#
#         cv2.fillPoly(color_warp, np.int_([pts]), (216, 168, 74))
#         cv2.fillPoly(color_warp, np.int_([pts_mean]), (216, 168, 74))
#
#         newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
#         result = cv2.addWeighted(original_image, 1, newwarp, 0.4, 0)
#
#     return pts_mean, result


#trial and error로 일단 제외된 함수들
# def color_filter(image):
#     hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
#
#     lower = np.array([20, 150, 20])
#     upper = np.array([255, 255, 255])
#
#     yellow_lower = np.array([0, 85, 81])
#     yellow_upper = np.array([60, 255, 255])
#
#     yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
#     white_mask = cv2.inRange(hls, lower, upper)
#     mask = cv2.bitwise_or(yellow_mask, white_mask)
#     masked = cv2.bitwise_and(image, image, mask = mask)
#
#     return masked
# def schannel(image):
#     hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
#     h,l,s = cv2.split(hls)
#     return s
# def thresholding(s_channel, scaled_sobel, s_thresh=(170, 255), sx_thresh=(20, 100)):
#     # Threshold x gradient
#     sxbinary = np.zeros_like(scaled_sobel)
#     sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
#     # Threshold color channel
#     s_binary = np.zeros_like(s_channel)
#     s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
#
#     combined_binary = cv2.bitwise_or(s_binary, sxbinary)
#     return  combined_binary


fname = 'E:\\python_coding\\2021_summer_intern\\ObjectDetection\\CJtest\\Videos\\line.mp4'
capture = cv2.VideoCapture(fname)

#video should manage with frame-by-frame method
while True:
    isTrue, frame = capture.read()
    #preprocessing
    src, dst = image_warp_mtx(frame)
    # sobelx=sobel(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    canny = cv2.Canny(gray, 100, 150)
    binary_warped, M, Minv = perspect(canny, src, dst)
    ret, thresh = cv2.threshold(binary_warped, 60, 255, cv2.THRESH_BINARY)
    left, right= plothistogram(thresh)
    draw=window_search(thresh, left, right)
    # meanPts, result = draw_lane_lines(frame, thresh, Minv, draw)

    # #finding line
    # cv2.imshow('result', result)

    # stop triggerd
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()