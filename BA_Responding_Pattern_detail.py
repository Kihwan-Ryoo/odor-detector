import threading

import matplotlib.colors
import numpy as np
import pandas as pd
from random import *
import time
from cv2 import cv2
import tracktor as tr
from scipy.spatial.distance import cdist
import matplotlib
from matplotlib import cm


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
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                    None, scale, scale)
                    if len(imgArray[x][y].shape) == 2:
                        imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
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
            cv2.circle(final, (point[0], point[1]), 3, myColorValues[point[2]], cv2.FILLED)

    def drawColormap_jet(self, width, height, min, max):
        map = np.ones((height, width + 100, 3), np.uint8)
        map = map * 255
        norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        for i in range(width):
            for k in range(int((height / (max - min)) * 10)):
                rgb = cm.jet(norm((max - min) * k / ((height / (max - min)) * 10)), bytes=True)
                cv2.line(map, (i, int((max - min) * (((height / (max - min)) * 10 - k) / 10))), (i, int((max - min) * (((height / (max - min)) * 10 - (k + 1)) / 10))), tuple((int(rgb[2]), int(rgb[1]), int(rgb[0]))), 1)
        cv2.rectangle(map, (0, 0), (width - 1, height - 1), (0, 0, 0), 1)
        cv2.putText(map, '23 mm/s', (25, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(map, '0 mm/s', (25, height - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        return map

    def run(self):
        sec_interest = 100
        interest_start = 0

        dur_cleaning = 26  # ?
        dur_puff = 10
        dur_pre_puff = 15
        dur_post_puff_light_off = 10
        dur_post_puff_light_on = 20
        dur_post_puff_light_on_air = 10
        Block_points = np.zeros((5, 5, sec_interest - interest_start))
        for i in range(5):
            for j in range(5):
                Block_points[i][j][0] = ((dur_pre_puff + dur_puff + dur_post_puff_light_off + dur_post_puff_light_on + dur_post_puff_light_on_air) * 5 * i + (dur_cleaning + dur_pre_puff) + 65 * j) * 10 + interest_start
                for k in range(sec_interest - interest_start):
                    Block_points[i][j][k] = Block_points[i][j][0] + k
        print(Block_points)
        maxCounting_points = np.zeros((5, 5, 100))
        for i in range(5):
            for j in range(5):
                maxCounting_points[i][j][0] = ((dur_pre_puff + dur_puff + dur_post_puff_light_off + dur_post_puff_light_on + dur_post_puff_light_on_air) * 5 * i + (dur_cleaning + 5) + 65 * j) * 10 + interest_start
                for k in range(100):
                    maxCounting_points[i][j][k] = maxCounting_points[i][j][0] + k
        print(maxCounting_points)

        odor_sequence = pd.read_csv('output/20220120/Day/Cartridge1_odor.csv', header=None)
        print('odor' + str(odor_sequence[1][0]))
        print(odor_sequence)
        print(odor_sequence[0][0])
        odor_num = np.zeros(6, dtype=int)

        n_inds = 400
        colours = []
        for cnum in range(n_inds):
            colours.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
        colorbar_jet = self.drawColormap_jet(20, 800, 0, 20)
        Points = []
        # Points_cartridge = []
        # Points_central = []
        xy_collection = []
        last_xy_collection = []
        xy_collection_central = [[], [], [], [], [], []]
        last_xy_collection_central = [[], [], [], [], [], []]
        dist_thres = 20

        scaling = 1

        block_size = int(25 * scaling / 2) * 2 + 1  # 51
        offset = int(25 * scaling)

        min_area = int(18 * scaling) # 10 -- minimum fly size
        max_area = int(200 * scaling) # 300 -- maximum fly size

        # mask area
        # x
        topleft_x = int(240 * scaling)
        # y
        topleft_y = int(59 * scaling)
        # w
        bottomright_x = int(1350 * scaling)
        # h
        bottomright_y = int(950 * scaling)

        w = bottomright_x - topleft_x - 10 - 6
        h = bottomright_y - topleft_y

        xnum = 10
        ynum = 6
        xborders = list(np.linspace(0, bottomright_x - topleft_x, (xnum + 1)))
        yborders = list(np.linspace(0, bottomright_y - topleft_y, (ynum + 1)))

        X = ((bottomright_x - topleft_x) / 2)
        Y = ((bottomright_y - topleft_y) / ynum)

        mot = True

        cap = cv2.VideoCapture('output/20220120/Day/Cartridge1_W1118_W1118_W1118_W1118_W1118_Or7a_D_DPG_1-heptanol_2-pentanol_1AA_BA/t20220120_182854.mp4')
        if not cap.isOpened():
            raise Exception("Could not open the video device")

        meas_last = list(np.zeros((n_inds, 2)))
        meas_now = list(np.zeros((n_inds, 2)))
        df = []
        max_num = np.zeros(6)

        # FPS = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        frame_num = 0
        num_imageStacked = 0 # imageStacked 저장할 상태인지의 여부
        central_initial = np.zeros((2, n_inds))
        count_initial = [[], [], [], [], [], []]
        count_ongoing = [[], [], [], [], [], []]
        lostPoints = []
        while True:
            ret, frame = cap.read()
            frame_idx += 1
            if ret is True:
                # cartridge = np.ones(((bottomright_y - topleft_y + 50), (bottomright_x - topleft_x), 3), np.uint8)
                # cartridge = cartridge * 255
                # for i in range(ynum + 1):
                #     cv2.line(cartridge, (0, int(yborders[i])), (bottomright_x - topleft_x, int(yborders[i])), (0, 0, 0), 7)
                # for i in range(xnum - 1):
                #     cv2.line(cartridge, (int(xborders[i + 1]), 0), (int(xborders[i + 1]), bottomright_y - topleft_y), (0, 0, 0), 1)
                # cv2.line(cartridge, (0, 0), (0, bottomright_y - topleft_y), (0, 0, 0), 7)
                # cv2.line(cartridge, (bottomright_x - topleft_x, 0), (bottomright_x - topleft_x, bottomright_y - topleft_y), (0, 0, 0), 7)

                central = np.ones(((bottomright_y - topleft_y + 50), (bottomright_x - topleft_x), 3), np.uint8)
                central = central * 255
                cv2.line(central, (int(X), 0), (int(X), (bottomright_y - topleft_y)), (0, 0, 255), 1)
                for i in range(ynum + 1):
                    cv2.line(central, (0, int(yborders[i])), (bottomright_x - topleft_x, int(yborders[i])), (0, 0, 0), 7)
                cv2.line(central, (0, 0), (0, bottomright_y - topleft_y), (0, 0, 0), 7)
                cv2.line(central, (bottomright_x - topleft_x, 0), (bottomright_x - topleft_x, bottomright_y - topleft_y), (0, 0, 0), 7)
                central[50:850, 980:1000] = colorbar_jet[:, 0:20]
                central[50:80, 1000:1100] = colorbar_jet[0:30, 20:120]
                central[800:850, 1000:1100] = colorbar_jet[750:800, 20:120]

                # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # frame = cv2.resize(frame, (0, 0), fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
                bg = frame.copy()
                mask = np.zeros((frame.shape[0], frame.shape[1]))
                cv2.rectangle(mask, (topleft_x + 10, topleft_y), (bottomright_x - 6, bottomright_y), 255, -1)
                cv2.rectangle(mask, (bottomright_x - 15, bottomright_y - 150), (bottomright_x - 6, bottomright_y), 0, -1)
                frame[mask == 0] = 0

                thresh = tr.colour_to_thresh(frame, block_size, offset)
                final, contours, meas_last, meas_now = tr.detect_and_draw_contours(frame, thresh, meas_last, meas_now, min_area, max_area)

                # yx sorting
                meas_last = list(meas_last)
                meas_now = list(meas_now)
                meas_now.sort(key=lambda x:x[1])
                meas_now.sort(key=lambda x:x[0])

                row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
                final, meas_now, df = tr.reorder_and_draw(final, colours, n_inds, col_ind, meas_now, df, mot)
                final[mask == 0] = bg[mask == 0]
                for i in range(ynum + 1):
                    cv2.line(final, (topleft_x + 10, int(topleft_y + yborders[i])), (bottomright_x - 6, int(topleft_y + yborders[i])), (0, 0, 255), 1)

                if frame_idx in maxCounting_points:
                    num = np.zeros(ynum)
                    max_num = np.zeros(ynum)
                    meas_now = np.array(meas_now)
                    for i in range(ynum):
                        num[i] = np.sum(1 * ((meas_now[:, [1]] >= (yborders[i] + topleft_y)) & (meas_now[:, [1]] < (yborders[i + 1] + topleft_y))))
                    for i in range(ynum):
                        if max_num[i] < num[i]:
                            max_num[i] = num[i]
                    meas_now = list(meas_now)

                if frame_idx in Block_points:
                    count = 0
                    newPoints = []
                    # newPoints_cartridge = []
                    newPoints_central = [[], [], [], [], [], []]
                    newPoints_last_central = [[], [], [], [], [], []]
                    new_xy = []
                    new_last_xy = []
                    if frame_num == 1:
                        print('max_num', max_num)
                        print('count_initial', count_initial)
                        print(' ', str(len(count_initial[0])) + ' ' + str(len(count_ongoing[1])) + ' ' + str(len(count_initial[2])) + ' ' + str(len(count_initial[3])) + ' ' + str(len(count_initial[4])) + ' ' + str(len(count_initial[5])))
                    if count == 0:
                        # print('count_ongoing', count_ongoing)
                        print(' ', str(len(count_ongoing[0])) + ' ' + str(len(count_ongoing[1])) + ' ' + str(len(count_ongoing[2])) + ' ' + str(len(count_ongoing[3])) + ' ' + str(len(count_ongoing[4])) + ' ' + str(len(count_ongoing[5])))
                        count_ongoing = [[], [], [], [], [], []]
                    for i in range(len(meas_now)):
                        x = tuple([int(x) for x in meas_now[i]])[0]
                        y = tuple([int(x) for x in meas_now[i]])[1]
                        j = i
                        if j < len(meas_last):
                            last_x = tuple([int(x) for x in meas_last[j]])[0]
                            last_y = tuple([int(x) for x in meas_last[j]])[1]
                        else:
                            last_x, last_y = x, y
                        dm = [[last_x, last_y], [x, y]]
                        dist = cdist(dm, dm)[0][1]
                        line_color = cm.jet(norm(dist), bytes=True)
                        if frame_num == 0:  ###초기(t=0)값.설정. x,y 또는 last_x, last_y (t=0일때의 시작점과 관련됨)
                            central_initial[0][i], central_initial[1][i] = last_x, last_y
                            if ((last_y - topleft_y) >= yborders[0]) and ((last_y - topleft_y) < yborders[1]):
                                count_initial[0].append(count)
                            elif ((last_y - topleft_y) >= yborders[1]) and ((last_y - topleft_y) < yborders[2]):
                                count_initial[1].append(count)
                            elif ((last_y - topleft_y) >= yborders[2]) and ((last_y - topleft_y) < yborders[3]):
                                count_initial[2].append(count)
                            elif ((last_y - topleft_y) >= yborders[3]) and ((last_y - topleft_y) < yborders[4]):
                                count_initial[3].append(count)
                            elif ((last_y - topleft_y) >= yborders[4]) and ((last_y - topleft_y) < yborders[5]):
                                count_initial[4].append(count)
                            elif ((last_y - topleft_y) >= yborders[5]) and ((last_y - topleft_y) < yborders[6]):
                                count_initial[5].append(count)
                            first_num = str(len(count_initial[0])) + ' ' + str(len(count_initial[1])) + ' ' + str(len(count_initial[2])) + ' ' + str(len(count_initial[3])) + ' ' + str(len(count_initial[4])) + ' ' + str(len(count_initial[5]))
                        if ((y - topleft_y) >= yborders[0]) and ((y - topleft_y) < yborders[1]):
                            count_ongoing[0].append(count)
                        elif ((y - topleft_y) >= yborders[1]) and ((y - topleft_y) < yborders[2]):
                            count_ongoing[1].append(count)
                        elif ((y - topleft_y) >= yborders[2]) and ((y - topleft_y) < yborders[3]):
                            count_ongoing[2].append(count)
                        elif ((y - topleft_y) >= yborders[3]) and ((y - topleft_y) < yborders[4]):
                            count_ongoing[3].append(count)
                        elif ((y - topleft_y) >= yborders[4]) and ((y - topleft_y) < yborders[5]):
                            count_ongoing[4].append(count)
                        elif ((y - topleft_y) >= yborders[5]) and ((y - topleft_y) < yborders[6]):
                            count_ongoing[5].append(count)
                        if x != 0 and y != 0:
                            newPoints.append([x, y, count])
                            # newPoints_cartridge.append([x - topleft_x, y - topleft_y, count])
                            a, b = central_initial[0][i], central_initial[1][i]
                            new_xy.append([x, y, count])
                            new_last_xy.append([last_x, last_y, count])
                            if a != 0 and b != 0:
                                if dist < dist_thres:
                                    if ((y - topleft_y) >= yborders[0]) and ((y - topleft_y) < yborders[1]) and (count in count_initial[0]):
                                        newPoints_central[0].append([int(x - topleft_x + X - (a - topleft_x)), int(y - topleft_y + (Y * (0.5 + 0)) - (b - topleft_y)), count])
                                    elif ((y - topleft_y) >= yborders[1]) and ((y - topleft_y) < yborders[2]) and (count in count_initial[1]):
                                        newPoints_central[1].append([int(x - topleft_x + X - (a - topleft_x)), int(y - topleft_y + (Y * (0.5 + 1)) - (b - topleft_y)), count])
                                    elif ((y - topleft_y) >= yborders[2]) and ((y - topleft_y) < yborders[3]) and (count in count_initial[2]):
                                        newPoints_central[2].append([int(x - topleft_x + X - (a - topleft_x)), int(y - topleft_y + (Y * (0.5 + 2)) - (b - topleft_y)), count])
                                    elif ((y - topleft_y) >= yborders[3]) and ((y - topleft_y) < yborders[4]) and (count in count_initial[3]):
                                        newPoints_central[3].append([int(x - topleft_x + X - (a - topleft_x)), int(y - topleft_y + (Y * (0.5 + 3)) - (b - topleft_y)), count])
                                    elif ((y - topleft_y) >= yborders[4]) and ((y - topleft_y) < yborders[5]) and (count in count_initial[4]):
                                        newPoints_central[4].append([int(x - topleft_x + X - (a - topleft_x)), int(y - topleft_y + (Y * (0.5 + 4)) - (b - topleft_y)), count])
                                    elif ((y - topleft_y) >= yborders[5]) and ((y - topleft_y) < yborders[6]) and (count in count_initial[5]):
                                        newPoints_central[5].append([int(x - topleft_x + X - (a - topleft_x)), int(y - topleft_y + (Y * (0.5 + 5)) - (b - topleft_y)), count])
                                    else:
                                        for i in range(6):
                                            if count in count_initial[i]:
                                                count_initial[i].remove(count)
                                else:
                                    for i in range(6):
                                        if count in count_initial[i]:
                                            count_initial[i].remove(count)

                                if dist < dist_thres:
                                    if ((last_y - topleft_y) >= yborders[0]) and ((last_y - topleft_y) < yborders[1]) and (count in count_initial[0]):
                                        newPoints_last_central[0].append([int(last_x - topleft_x + X - (a - topleft_x)), int(last_y - topleft_y + (Y * (0.5 + 0)) - (b - topleft_y)), count, line_color[2], line_color[1], line_color[0]])
                                    elif ((last_y - topleft_y) >= yborders[1]) and ((last_y - topleft_y) < yborders[2]) and (count in count_initial[1]):
                                        newPoints_last_central[1].append([int(last_x - topleft_x + X - (a - topleft_x)), int(last_y - topleft_y + (Y * (0.5 + 1)) - (b - topleft_y)), count, line_color[2], line_color[1], line_color[0]])
                                    elif ((last_y - topleft_y) >= yborders[2]) and ((last_y - topleft_y) < yborders[3]) and (count in count_initial[2]):
                                        newPoints_last_central[2].append([int(last_x - topleft_x + X - (a - topleft_x)), int(last_y - topleft_y + (Y * (0.5 + 2)) - (b - topleft_y)), count, line_color[2], line_color[1], line_color[0]])
                                    elif ((last_y - topleft_y) >= yborders[3]) and ((last_y - topleft_y) < yborders[4]) and (count in count_initial[3]):
                                        newPoints_last_central[3].append([int(last_x - topleft_x + X - (a - topleft_x)), int(last_y - topleft_y + (Y * (0.5 + 3)) - (b - topleft_y)), count, line_color[2], line_color[1], line_color[0]])
                                    elif ((last_y - topleft_y) >= yborders[4]) and ((last_y - topleft_y) < yborders[5]) and (count in count_initial[4]):
                                        newPoints_last_central[4].append([int(last_x - topleft_x + X - (a - topleft_x)), int(last_y - topleft_y + (Y * (0.5 + 4)) - (b - topleft_y)), count, line_color[2], line_color[1], line_color[0]])
                                    elif ((last_y - topleft_y) >= yborders[5]) and ((last_y - topleft_y) < yborders[6]) and (count in count_initial[5]):
                                        newPoints_last_central[5].append([int(last_x - topleft_x + X - (a - topleft_x)), int(last_y - topleft_y + (Y * (0.5 + 5)) - (b - topleft_y)), count, line_color[2], line_color[1], line_color[0]])
                                    else:
                                        for i in range(6):
                                            if count in count_initial[i]:
                                                count_initial[i].remove(count)
                                else:
                                    for i in range(6):
                                        if count in count_initial[i]:
                                            count_initial[i].remove(count)
                        count += 1
                    if len(newPoints) != 0:
                        for newP in newPoints:
                            Points.append(newP)
                        # for newP_cartridge in newPoints_cartridge:
                        #     Points_cartridge.append(newP_cartridge)
                        for i in range(6):
                            if len(newPoints_central[i]) != 0:
                                for newP_central in newPoints_central[i]:
                                    # Points_central.append(newP_central)
                                    xy_collection_central[i].append(newP_central)
                                for newP_last_central in newPoints_last_central[i]:
                                    last_xy_collection_central[i].append(newP_last_central)
                        for xy in new_xy:
                            xy_collection.append(xy)
                        for last_xy in new_last_xy:
                            last_xy_collection.append(last_xy)
                    frame_num += 1
                    if (len(Points) != 0) and (frame_num == sec_interest - interest_start):
                        last_num = str(len(count_initial[0])) + ' ' + str(len(count_initial[1])) + ' ' + str(len(count_initial[2])) + ' ' + str(len(count_initial[3])) + ' ' + str(len(count_initial[4])) + ' ' + str(len(count_initial[5]))
                        self.drawOnCanvas(final, Points, colours)
                        # self.drawOnCanvas(cartridge, Points_cartridge, colours)
                        #self.drawOnCanvas(central, Points_central, colours)
                        for xy in range(len(xy_collection)):
                            cv2.line(final, (last_xy_collection[xy][0], last_xy_collection[xy][1]), (xy_collection[xy][0], xy_collection[xy][1]), (0, 0, 0), 1)
                            # cv2.line(cartridge, (last_xy_collection[xy][0] - topleft_x, last_xy_collection[xy][1] - topleft_y), (xy_collection[xy][0] - topleft_x, xy_collection[xy][1] - topleft_y), (0, 0, 0), 1)
                            for n in range(6):
                                last_xy_collection_central[n].sort(key=lambda x:x[2])
                                xy_collection_central[n].sort(key=lambda x:x[2])
                                for xy_central in range(len(xy_collection_central[n])):
                                    if xy_central < len(last_xy_collection_central[n]):
                                        if last_xy_collection_central[n][xy_central][2] == xy_collection_central[n][xy_central][2]:
                                            color = (int(last_xy_collection_central[n][xy_central][3]), int(last_xy_collection_central[n][xy_central][4]), int(last_xy_collection_central[n][xy_central][5]))
                                            cv2.line(central, (last_xy_collection_central[n][xy_central][0], last_xy_collection_central[n][xy_central][1]), (xy_collection_central[n][xy_central][0], xy_collection_central[n][xy_central][1]), tuple(color), 1)
                        situation = '_(w,h,block size,offset,min area)_(' + str(w) + ',' + str(h) + ',' + str(block_size) + ',' + str(offset) + ',' + str(min_area) + ')'
                        cv2.putText(final, first_num, (20, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                        cv2.putText(final, last_num, (20, 200), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                        cv2.putText(final, 't20220120_182854_odor' + str(odor_sequence[odor_num[0]][0]) + '-' + str(odor_num[odor_sequence[odor_num[0]][0]] + 1) + '_' + str(int(interest_start / 10)) + 's_to_' + str(int(sec_interest / 10)) + 's', (220, 1000), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
                        cv2.imwrite('output/20220120/Day/Cartridge1_W1118_W1118_W1118_W1118_W1118_Or7a_D_DPG_1-heptanol_2-pentanol_1AA_BA/detail_result/t20220120_182854_odor' + str(odor_sequence[odor_num[0]][0]) + '-' + str(odor_num[odor_sequence[odor_num[0]][0]] + 1) + '_' + str(int(interest_start / 10)) + 's_to_' + str(int(sec_interest / 10)) + 's' + situation + '.png', final)
                        # cv2.putText(cartridge, '[white]t20220120_182854_odor' + str(odor_sequence[odor_num[0]][0]) + '-' + str(odor_num[odor_sequence[odor_num[0]][0]] + 1) + '_' + str(int(interest_start / 10)) + 's_to_' + str(int(sec_interest / 10)) + 's', (15, 935), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                        # cv2.imwrite('output/20220120/Day/Cartridge1_W1118_W1118_W1118_W1118_W1118_Or7a_D_DPG_1-heptanol_2-pentanol_1AA_BA/detail_result/[white]t20220120_182854_odor' + str(odor_sequence[odor_num[0]][0]) + '-' + str(odor_num[odor_sequence[odor_num[0]][0]] + 1) + '_' + str(int(interest_start / 10)) + 's_to_' + str(int(sec_interest / 10)) + 's' + situation + '.png', cartridge)
                        cv2.putText(central, '[central]t20220120_182854_odor' + str(odor_sequence[odor_num[0]][0]) + '-' + str(odor_num[odor_sequence[odor_num[0]][0]] + 1) + '_' + str(int(interest_start / 10)) + 's_to_' + str(int(sec_interest / 10)) + 's', (15, 930), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
                        cv2.imwrite('output/20220120/Day/Cartridge1_W1118_W1118_W1118_W1118_W1118_Or7a_D_DPG_1-heptanol_2-pentanol_1AA_BA/detail_result/[central]t20220120_182854_odor' + str(odor_sequence[odor_num[0]][0]) + '-' + str(odor_num[odor_sequence[odor_num[0]][0]] + 1) + '_' + str(int(interest_start / 10)) + 's_to_' + str(int(sec_interest / 10)) + 's' + situation + '.png', central)
                        num_imageStacked = 1
                        num_imageStacked_odor_sequence = odor_sequence[odor_num[0]][0]
                        num_imageStacked_odor_num = odor_num[odor_sequence[odor_num[0]][0]]
                        odor_num[odor_sequence[odor_num[0]][0]] += 1
                        odor_num[0] += 1
                        max_num = np.zeros(ynum)
                        frame_num = 0
                        Points = []
                        # Points_cartridge = []
                        # Points_central = []
                        xy_collection = []
                        last_xy_collection = []
                        xy_collection_central = [[], [], [], [], [], []]
                        last_xy_collection_central = [[], [], [], [], [], []]
                        central_initial = np.zeros((2, n_inds))
                        count_initial = [[], [], [], [], [], []]
                        lostPoints = []

                imgStacked = self.stackImages(0.8, ([final, central]))
                if num_imageStacked == 1:
                    cv2.imwrite('output/20220120/Day/Cartridge1_W1118_W1118_W1118_W1118_W1118_Or7a_D_DPG_1-heptanol_2-pentanol_1AA_BA/detail_result/[dual]t20220120_182854_odor' + str(num_imageStacked_odor_sequence) + '-' + str(num_imageStacked_odor_num + 1) + '_' + str(int(interest_start / 10)) + 's_to_' + str(int(sec_interest / 10)) + 's' + situation + '.png', imgStacked)
                    num_imageStacked = 0
                cv2.imshow("W1118(5) + Or7a(1)", imgStacked)
                if cv2.waitKey(10) & 0xff == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


# t1 = odorThread()
t2 = videoThread()

# t1.start()
t2.start()