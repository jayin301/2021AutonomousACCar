import numpy as np
import cv2


class Line:
    def __init__(self):
        self.detected = False
        self.window_margin = 56
        self.prevx = []
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.startx = None
        self.endx = None
        self.allx = None
        self.ally = None
        self.curvature = None
        self.steering = None

def warp_image(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)

    return warp_img, M, Minv

def find_LR_lines(binary_img, center_line):
    if center_line.detected == False:
        return blind_search(binary_img,center_line)
    else:
        return prev_window_refer(binary_img, center_line)

def rad_of_curvature(center_line):
    ploty = center_line.ally
    centerx = center_line.allx
    centerx = centerx[::-1]

    """
    ymperpix, xmperpix 고치기
    """
    # # Define conversions in x and y from pixels space to meters
    # # width_lanes = abs(right_line.startx - left_line.startx)
    # # ym_per_pix = 30 / 720  # parameter 교체
    # # xm_per_pix = 3.7*(720/1280) / width_lanes  #parameter 교체
    #
    # y_eval = np.max(ploty)
    # center_fit_cr = np.polyfit(ploty * ym_per_pix, centerx * xm_per_pix, 2)
    # center_curverad = ((1 + (2 * center_fit_cr[0] * y_eval * ym_per_pix + center_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
    #     2 * center_fit_cr[0])
    # center_line.radius_of_curvature = center_curverad

def smoothing(lines, pre_lines=3):
    lines = np.squeeze(lines)
    avg_line = np.zeros((720))

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:
            break
        avg_line += line
    avg_line = avg_line / pre_lines

    return avg_line

def blind_search(b_img, center_line):
    histogram = np.sum(b_img[int(b_img.shape[0] / 2):, :], axis=0)

    output = np.dstack((b_img, b_img, b_img)) * 255

    start_X = np.argmax(histogram[:])

    num_windows = 9
    window_height = np.int(b_img.shape[0] / num_windows)

    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    current_startX = start_X
    min_num_pixel = 50

    win_center_lane = []

    window_margin = center_line.window_margin

    for window in range(num_windows):
        win_y_low = b_img.shape[0] - (window + 1) * window_height
        win_y_high = b_img.shape[0] - window * window_height
        win_centerx_min = current_startX - window_margin
        win_centerx_max = current_startX + window_margin

        cv2.rectangle(output, (win_centerx_min, win_y_low), (win_centerx_max, win_y_high), (0, 255, 0), 2)

        center_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_centerx_min) & (
            nonzerox <= win_centerx_max)).nonzero()[0]
        win_center_lane.append(center_window_inds)

        if len(center_window_inds) > min_num_pixel:
            current_centerX = np.int(np.mean(nonzerox[center_window_inds]))

    win_center_lane = np.concatenate(win_center_lane)

    centerx, centery = nonzerox[win_center_lane], nonzeroy[win_center_lane]

    output[centery, centerx] = [255, 0, 0]

    center_fit = np.polyfit(centery, centerx, 2)

    center_line.current_fit = center_fit

    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    center_plotx = center_fit[0] * ploty ** 2 + center_fit[1] * ploty + center_fit[2]

    center_line.prevx.append(center_plotx)

    if len(center_line.prevx) > 10:
        center_avg_line = smoothing(center_line.prevx, 10)
        center_avg_fit = np.polyfit(ploty, center_avg_line, 2)
        center_fit_plotx = center_avg_fit[0] * ploty ** 2 + center_avg_fit[1] * ploty + center_avg_fit[2]
        center_line.current_fit = center_avg_fit
        center_line.allx, center_line.ally = center_fit_plotx, ploty
    else:
        center_line.current_fit = center_fit
        center_line.allx, center_line.ally = center_plotx, ploty

    center_line.startx = center_line.allx[len(center_line.allx)-1]
    center_line.endx= center_line.allx[0]

    center_line.detected = True
    rad_of_curvature(center_line)
    return output

def prev_window_refer(b_img, center_line):
    output = np.dstack((b_img, b_img, b_img)) * 255

    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    window_margin = center_line.window_margin

    center_line_fit = center_line.current_fit
    centerx_min = center_line_fit[0] * nonzeroy ** 2 + center_line_fit[1] * nonzeroy + center_line_fit[2] - window_margin
    centerx_max = center_line_fit[0] * nonzeroy ** 2 + center_line_fit[1] * nonzeroy + center_line_fit[2] + window_margin

    center_inds = ((nonzerox >= centerx_min) & (nonzerox <= centerx_max)).nonzero()[0]

    centerx, centery = nonzerox[center_inds], nonzeroy[center_inds]

    output[centery, centerx] = [255, 0, 0]

    center_fit = np.polyfit(centery, centerx, 2)

    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    center_plotx = center_fit[0] * ploty ** 2 + center_fit[1] * ploty + center_fit[2]

    centerx_avg = np.average(center_plotx)

    center_line.prevx.append(center_plotx)

    if len(center_line.prevx) > 10:
        center_avg_line = smoothing(center_line.prevx, 10)
        center_avg_fit = np.polyfit(ploty, center_avg_line, 2)
        center_fit_plotx = center_avg_fit[0] * ploty ** 2 + center_avg_fit[1] * ploty + center_avg_fit[2]
        center_line.current_fit = center_avg_fit
        center_line.allx, center_line.ally = center_fit_plotx, ploty
    else:
        center_line.current_fit = center_fit
        center_line.allx, center_line.ally = center_plotx, ploty

    center_line.startx = center_line.allx[len(center_line.allx) - 1]
    center_line.endx = center_line.allx[0]

    rad_of_curvature(center_line)
    return output



# def draw_lane(img, left_line, right_line, lane_color=(255, 0, 255), road_color=(0, 255, 0)):
#     """ draw lane lines & current driving space """
#     window_img = np.zeros_like(img)
#
#     window_margin = left_line.window_margin
#     left_plotx, right_plotx = left_line.allx, right_line.allx
#     ploty = left_line.ally
#
#     # Generate a polygon to illustrate the search window area
#     # And recast the x and y points into usable format for cv2.fillPoly()
#     left_pts_l = np.array([np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
#     left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))])
#     left_pts = np.hstack((left_pts_l, left_pts_r))
#     right_pts_l = np.array([np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))])
#     right_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))])
#     right_pts = np.hstack((right_pts_l, right_pts_r))
#
#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
#     cv2.fillPoly(window_img, np.int_([right_pts]), lane_color)
#
#     # Recast the x and y points into usable format for cv2.fillPoly()
#     pts_left = np.array([np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
#     pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx-window_margin/5, ploty])))])
#     pts = np.hstack((pts_left, pts_right))
#
#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([pts]), road_color)
#     result = cv2.addWeighted(img, 1, window_img, 0.3, 0)
#
#     return result, window_img
#
# def road_info(left_line, right_line):
#     """ print road information onto result image """
#     curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2
#     cm_curvature = curvature*100
#     steering = 50.353*pow(cm_curvature,-0.97)
#     direction = ((left_line.endx - left_line.startx) + (right_line.endx - right_line.startx)) / 2
#
#     if curvature > 2000 and abs(direction) < 100:
#         road_inf = 'No Curve'
#         curvature = -1
#     elif curvature <= 2000 and direction < - 50:
#         road_inf = 'Left Curve'
#         steering = steering * -1
#     elif curvature <= 2000 and direction > 50:
#         road_inf = 'Right Curve'
#     else:
#         if left_line.road_inf != None:
#             road_inf = left_line.road_inf
#             curvature = left_line.curvature
#         else:
#             road_inf = 'None'
#             curvature = curvature
#
#     center_lane = (right_line.startx + left_line.startx) / 2
#     lane_width = right_line.startx - left_line.startx
#
#     center_car = 720 / 2
#     if center_lane > center_car:
#         deviation = 'Left ' + str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
#     elif center_lane < center_car:
#         deviation = 'Right ' + str(round(abs(center_lane - center_car)/(lane_width / 2)*100, 3)) + '%'
#     else:
#         deviation = 'Center'
#
#     steering = str(round(steering, 3))
#
#     left_line.road_inf = road_inf
#     left_line.curvature = curvature
#     left_line.deviation = deviation
#     left_line.steering = steering
#
#     return road_inf, curvature, deviation, steering
#
# def print_road_status(img, left_line, right_line):
#     """ print road status (curve direction, radius of curvature, deviation) """
#     road_inf, curvature, deviation, steering = road_info(left_line, right_line)
#
#     cv2.putText(img, 'Road Status', (22, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (80, 80, 80), 2)
#
#     lane_inf = 'Lane Info : ' + road_inf
#     if curvature == -1:
#         lane_curve = 'Curvature : Straight line'
#     else:
#         lane_curve = 'Curvature : {0:0.3f}m'.format(curvature)
#     steer = 'Steer Angle : '+ steering
#
#     cv2.putText(img, lane_inf, (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
#     cv2.putText(img, lane_curve, (10, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
#     cv2.putText(img, steer, (10, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)
#
#     return img