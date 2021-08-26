import numpy as np
import cv2
from threshold import gradient_combine, hls_combine, comb_result
from finding_lines import Line, warp_image, find_LR_lines

input_name = 'project.mp4'

center_line = Line()

th_sobelx, th_sobely, th_mag, th_dir = (40, 100), (60, 255), (150, 255), (0.7, 1.3)
th_h, th_l, th_s = (20, 40), (0, 60), (85, 255)

if __name__ == '__main__':
    cap = cv2.VideoCapture(input_name)
    while (cap.isOpened()):
        _, frame = cap.read()
        img = cv2.resize(frame, dsize = (640, 480), interpolation=cv2.INTER_AREA)
        undist_img = cv2.GaussianBlur(img, (5,5),0)
        rows, cols = undist_img.shape[:2]

        combined_gradient = gradient_combine(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
        # cv2.imshow('gradient combined image', combined_gradient)
        combined_hls = hls_combine(undist_img, th_h, th_l, th_s)
        # cv2.imshow('HLS combined image', combined_hls)
        combined_result = comb_result(combined_gradient, combined_hls)
        cv2.imshow('comb_result', combined_result)

        c_rows, c_cols = combined_result.shape[:2]
        s_LTop2, s_RTop2 = [5, 5], [c_cols - 5, 5]
        s_LBot2, s_RBot2 = [5, c_rows], [c_cols - 5, c_rows]
        src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        dst = np.float32([(0, 720), (0, 0), (700, 0), (700, 720)])

        warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
        cv2.imshow('warp', warp_img)

        try:
            searching_img = find_LR_lines(warp_img, center_line)
        except TypeError:
            continue
        cv2.imshow('LR searching', searching_img)

        # w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)
        # #cv2.imshow('w_comb_result', w_comb_result)
        #
        # # Drawing the lines back down onto the road
        # color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
        # lane_color = np.zeros_like(undist_img)
        # lane_color[220:rows - 12, 0:cols] = color_result
        #
        # # Combine the result with the original image
        # result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)
        # #cv2.imshow('result', result.astype(np.uint8))
        #
        # info= np.zeros_like(result)
        # info[5:140, 5:140] = (255, 255, 255)
        # info = cv2.addWeighted(result, 1, info, 0.2, 0)
        # road_map = print_road_map(w_color_result, left_line, right_line)
        # info2[10:105, cols-106:cols-11] = road_map
        # info2 = print_road_status(info2, left_line, right_line)
        # cv2.imshow('road info', info2)

        # out.write(frame)
        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

    cap.release()
    cv2.destroyAllWindows()

