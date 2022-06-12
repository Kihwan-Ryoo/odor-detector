import threading

import os
import sys
import numpy as np
import pandas as pd
from random import *
import time
from cv2 import cv2
import tracktor as tr

valve_status = 0

class odorThread(threading.Thread):
    def __int__(self):
        threading.Thread.__init__(self)


    def run(self):
        global valve_status

        dur_cleaning = 30
        dur_puff = 10
        dur_pre_puff = 15
        dur_post_puff_light_off = 10
        dur_post_puff_light_on = 20
        dur_post_puff_light_on_air = 10

        valve_status = 0
        time.sleep(dur_cleaning)

        num_blocks = 5
        for block_id in range(num_blocks):
            print("Block#" + str(block_id) + " start")
            for valve in range(5):
                time.sleep(dur_pre_puff)
                valve_status = 1
                print("An odor is ON")
                time.sleep(dur_puff)
                valve_status = 0
                print("The odor is OFF")
                time.sleep(dur_post_puff_light_off)
                valve_status = 10000
                time.sleep(dur_post_puff_light_on - dur_post_puff_light_on_air)
                time.sleep(dur_post_puff_light_on_air)
                valve_status = 0

        print(" test ended -- press q")
        return None


class videoThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)


    def stackImages(self, scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2:
                        imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0,0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
                if len(imgArray[x].shape) == 2:
                    imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver


    def empty(self, a):
        pass


    def drawOnCanvas(self, final, myPoints, myColorValues):
        for point in myPoints:
            cv2.circle(final, (point[0], point[1]), 1, myColorValues[point[2]], cv2.FILLED)


    def run(self):
        global valve_status
        dur_cleaning = 26 # ?
        dur_puff = 10
        dur_pre_puff = 15
        dur_post_puff_light_off = 10
        dur_post_puff_light_on = 20
        dur_post_puff_light_on_air = 10
        Block_points = np.zeros((5, 5, 100))
        for i in range(5):
            for j in range(5):
                Block_points[i][j][0] = ((dur_pre_puff + dur_puff + dur_post_puff_light_off + dur_post_puff_light_on + dur_post_puff_light_on_air) * 5 * i
                                         + (dur_cleaning + dur_pre_puff) + 65 * j) * 10
                for k in range(100):
                    Block_points[i][j][k] = Block_points[i][j][0] + k
        print(Block_points)

        odor_sequence = pd.read_csv('output/20220120/Day/Cartridge1_odor.csv', header=None)
        print('odor' + str(odor_sequence[1][0]))
        print(odor_sequence)
        print(odor_sequence[0][0])
        odor_num = np.zeros(6, dtype=int)

        n_inds = 300
        colours = []
        for cnum in range(n_inds):
            colours.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        Points = []

        scaling = 1

        block_size = int(31 * scaling / 2) * 2 + 1 # 51
        offset = int(25 * scaling)
        '''
        min_area = int(60 * scaling) # 10 -- minimum fly size
        max_area = int(200 * scaling) # 300 -- maximum fly size
        
        # mask area
        # x
        topleft_x = int(210 * scaling)
        # y
        topleft_y = int(59 * scaling)
        # w
        bottomright_x = int(1350 * scaling)
        # h
        bottomright_y = int(958 * scaling)
        '''
        cv2.namedWindow("TrackBars")
        cv2.resizeWindow("TrackBars", 640, 240)
        cv2.createTrackbar("topleft_x", "TrackBars", 240, 1000, self.empty)
        cv2.createTrackbar("topleft_y", "TrackBars", 59, 1000, self.empty)
        cv2.createTrackbar("bottomright_x", "TrackBars", 1350, 1360, self.empty)
        cv2.createTrackbar("bottomright_y", "TrackBars", 958, 1000, self.empty)
        cv2.createTrackbar("min_area", "TrackBars", 18, 100, self.empty)
        cv2.createTrackbar("max_area", "TrackBars", 200, 500, self.empty)

        mot = True

        cap = cv2.VideoCapture('output/20220120/Day/Cartridge1_W1118_W1118_W1118_W1118_W1118_Or7a_D_DPG_1-heptanol_2-pentanol_1AA_BA/t20220120_182854.mp4')
        if not cap.isOpened():
            raise Exception("Could not open the video device")

        meas_last = list(np.zeros((n_inds, 2)))
        meas_now = list(np.zeros((n_inds, 2)))
        df = []

        FPS = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        frame_num = 0
        while True:
            topleft_x = cv2.getTrackbarPos("topleft_x", "TrackBars")
            topleft_y = cv2.getTrackbarPos("topleft_y", "TrackBars")
            bottomright_x = cv2.getTrackbarPos("bottomright_x", "TrackBars")
            bottomright_y = cv2.getTrackbarPos("bottomright_y", "TrackBars")
            min_area = cv2.getTrackbarPos("min_area", "TrackBars")
            max_area = cv2.getTrackbarPos("max_area", "TrackBars")

            ret, frame = cap.read()
            frame_idx += 1
            if ret is True:
                #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = cv2.resize(frame, (0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
                bg = frame.copy()
                mask = np.zeros((frame.shape[0], frame.shape[1]))
                cv2.rectangle(mask, (topleft_x, topleft_y), (bottomright_x, bottomright_y), 255, -1)
                frame[mask == 0] = 0
                thresh = tr.colour_to_thresh(frame, block_size, offset)
                final, contours, meas_last, meas_now = tr.detect_and_draw_contours(frame, thresh, meas_last, meas_now, min_area, max_area)
                row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
                final, meas_now, df = tr.reorder_and_draw(final, colours, n_inds, col_ind, meas_now, df, mot)
                final[:, 0:240] = bg[:, 0:240]

                if frame_idx in Block_points:
                    temp = frame.copy()
                    count = 0
                    newPoints = []
                    for i in range(len(meas_now)):
                        x = tuple([int(x) for x in meas_now[i]])[0]
                        y = tuple([int(x) for x in meas_now[i]])[1]
                        if x != 0 and y != 0:
                            newPoints.append([x, y, count])
                        count += 1
                    if len(newPoints) != 0:
                        for newP in newPoints:
                            Points.append(newP)
                    if len(Points) != 0:
                        self.drawOnCanvas(final, Points, colours)
                    frame_num += 1
                    if frame_num == 100:
                        #cv2.imwrite('output/20220120/Day/Cartridge1_W1118_W1118_W1118_W1118_W1118_Or7a_D_DPG_1-heptanol_2-pentanol_1AA_BA/t20220120_182854_odor' + str(odor_sequence[odor_num[0]][0]) + '_' + str(odor_num[odor_sequence[odor_num[0]][0]]) + '.png', final)
                        odor_num[odor_sequence[odor_num[0]][0]] += 1
                        odor_num[0] += 1
                        frame_num = 0
                        Points = []
                        frame = temp

                imgStacked = self.stackImages(0.7, ([bg, final]))
                cv2.imshow("W1118(5) + Or7a(1)", final)
                if cv2.waitKey(1) and 0xff == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


t1 = odorThread()
t2 = videoThread()

#t1.start()
t2.start()