import numpy as np
import cv2
from PIL import Image
from matplotlib import cm
import os

class CarDetector(object):

    def __init__(self, show_bounding_box, debug=False):
        self.car_cascade = cv2.CascadeClassifier("/home/autocar2/vehicle_detection_haarcascades/car.xml")
        self.show_bounding_box = show_bounding_box
        self.debug = debug

    def convertImageArrayToPILImage(self, img_arr):
        img = Image.fromarray(img_arr.astype('uint8'), 'RGB')
        return img

    '''
    Return an object if there is a traffic light in the frame
    '''
    def detect_car (self, img_arr):
        # img = self.convertImageArrayToPILImage(img_arr)
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        cars = self.car_cascade.detectMultiScale(gray, 1.1, 6) # scale factor, minNeighbors

        # car_detected = None
        is_detected = False

        if cars != None:
            self.cars = cars
            is_detected = True

        # if car_detected:
        #     self.last_5_scores.append(car_detected.score)
        #     sum_of_last_5_score = sum(list(self.last_5_scores))
        #     # print("sum of last 5 score = ", sum_of_last_5_score)

        #     if sum_of_last_5_score > self.LAST_5_SCORE_THRESHOLD:
        #         return car_detected
        #     else:
        #         print("Not reaching last 5 score threshold")
        #         return None
        # else:
        #     self.last_5_scores.append(0)
        #     return None

        return is_detected

    def draw_bounding_box(self, cars, img_arr):
        for (x, y, w, h) in cars:
            cv2.rectangle(img_arr, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def run(self, img_arr, angle, throttle, debug=False):
        if img_arr is None:
            return angle, throttle, img_arr

        # Detect car
        is_detected = self.detect_car(img_arr)

        if is_detected:
            if self.show_bounding_box:
                self.draw_bounding_box(self.cars, img_arr)
            return angle, 0, img_arr
        else:
            return angle, throttle, img_arr
