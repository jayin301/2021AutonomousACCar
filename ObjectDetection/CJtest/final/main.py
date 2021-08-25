import numpy as np
import cv2
from threshold import gradient_combine, hls_combine, comb_result
from finding_lines import Line, warp_image, find_LR_lines, draw_lane, print_road_status, print_road_map

input_name = '1.mp4'
left_line = Line()
right_line = Line()

th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (60, 255)


if __name__ == '__main__':
    cap = cv2.VideoCapture(input_name)
    while (cap.isOpened()):
        _, frame = cap.read()

        frame_img = cv2.resize(frame, (640, 480),  interpolation=cv2.INTER_AREA)
        rows, cols = frame_img.shape[:2]

        combined_gradient = gradient_combine(frame_img, th_sobelx, th_sobely, th_mag, th_dir)
        # cv2.imshow('gradient combined image', combined_gradient)
        combined_hls = hls_combine(frame_img, th_h, th_l, th_s)
        # cv2.imshow('HLS combined image', combined_hls)
        combined_result = comb_result(combined_gradient, combined_hls)
        # cv2.imshow('result', combined_result)

        c_rows, c_cols = combined_result.shape[:2]
        #for video 1
        s_LTop2, s_RTop2 = [c_cols / 2 - 30, 60], [c_cols / 2 + 45, 60]
        s_LBot2, s_RBot2 = [c_cols * 0.4, c_rows], [c_cols*0.66, c_rows]

        # for video 2
        # s_LTop2, s_RTop2 = [c_cols / 2 - 30, 70], [c_cols / 2 + 45, 70]
        # s_LBot2, s_RBot2 = [c_cols * 0.4, c_rows], [c_cols*0.66, c_rows]

        # for video 3
        # s_LTop2, s_RTop2 = [c_cols / 2 - 40, 80], [c_cols / 2 + 70, 80]
        # s_LBot2, s_RBot2 = [c_cols * 0.35, c_rows], [c_cols*0.72, c_rows]

        # for video 4
        # s_LTop2, s_RTop2 = [c_cols / 2 - 40, 80], [c_cols / 2 + 70, 80]
        # s_LBot2, s_RBot2 = [c_cols * 0.35, c_rows], [c_cols*0.72, c_rows]

        src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

        warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
        # cv2.imshow('warp', warp_img)
        try:
            searching_img = find_LR_lines(warp_img, left_line, right_line)
        except TypeError:
            continue

        w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)
        # cv2.imshow('w_comb_result', w_comb_result)

        color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
        lane_color = np.zeros_like(frame_img)
        lane_color[200:rows - 110, 0:cols] = color_result

        result = cv2.addWeighted(frame_img, 1, lane_color, 0.3, 0)

        info, info2 = np.zeros_like(result),  np.zeros_like(result)
        info[5:140, 5:190] = (255, 255, 255)
        info2[5:110, cols-111:cols-6] = (255, 255, 255)
        info = cv2.addWeighted(result, 1, info, 0.2, 0)
        info2 = cv2.addWeighted(info, 1, info2, 0.2, 0)
        road_map = print_road_map(w_color_result, left_line, right_line)
        info2[10:105, cols-106:cols-11] = road_map
        info2 = print_road_status(info2, left_line, right_line)
        cv2.imshow('road info', info2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

