#
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from threading import Timer
import cv2
import tracktor as tr
from random import *
import numpy as np
import pandas as pd
from pyqtgraph.Qt import QtCore, QtGui
import time
import pyfirmata as pf
import serial
import serial.tools.list_ports
import sklearn.neighbors._partition_nodes
sys.modules['sklearn.neighbors._partition_nodes'] = sklearn.neighbors._partition_nodes

def get_ports():
    ports = serial.tools.list_ports.comports()
    return ports

def findArduino(portsFound):
    commPort = 'None'
    numConnection = len(portsFound)
    for i in range(0, numConnection):
        port = foundPorts[i]
        strPort = str(port)
        if 'Arduino' in strPort:
            splitPort = strPort.split(' ')
            commPort = (splitPort[0])
    return commPort

foundPorts = get_ports()
connectPort = findArduino(foundPorts)


class BRL_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #self.setupUi(self) # 해당 명령어로 인해 '불필요'하게 전체 과정이 2번씩 진행되어 문제가 생겼었다.
        self.ard = pf.Arduino(connectPort)
        self.ID_LIGHT = 13
        self.ID_VACUUM = 12

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowIcon(QIcon("C:/Users/RKH/PycharmProjects/FlyCV/Resources/flies.png"))
        self.lane_names = ['W1118', 'W1118', 'W1118', 'W1118', 'W1118', 'Or7a']
        self.size = 2
        MainWindow.resize(int(960 * self.size), int(540 * self.size))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")


        # Camera
        self.frame_camera = QtWidgets.QFrame(self.centralwidget)
        self.frame_camera.setGeometry(QtCore.QRect(int(0 * self.size), int(0 * self.size), int(320 * self.size), int(320 * self.size)))
        self.frame_camera.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_camera.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_camera.setObjectName("frame_camera")

        self.pushButton_camera = QtWidgets.QPushButton(self.frame_camera)
        self.pushButton_camera.setGeometry(QtCore.QRect(int(110 * self.size), int(5 * self.size), int(100 * self.size), int(30 * self.size)))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_camera.setFont(font)
        self.pushButton_camera.setObjectName("pushButton_camera")

        self.label_camera = QLabel(self.frame_camera)
        self.label_camera.setGeometry(QRect(int(10 * self.size), int(40 * self.size), int(300 * self.size), int(270 * self.size)))
        self.label_camera.setObjectName("label_camera")


        self.topleft_x = 240
        self.topleft_y = 59
        self.bottomright_x = 1350
        self.bottomright_y = 958
        self.min_area = 18
        self.max_area = 200


        # 2D
        self.frame_2d = QtWidgets.QFrame(self.centralwidget)
        self.frame_2d.setGeometry(QtCore.QRect(int(320 * self.size), int(0 * self.size), int(320 * self.size), int(320 * self.size)))
        self.frame_2d.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2d.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2d.setObjectName("frame_2d")

        self.pushButton_2d = QtWidgets.QPushButton(self.frame_2d)
        self.pushButton_2d.setGeometry(QtCore.QRect(int(110 * self.size), int(5 * self.size), int(100 * self.size), int(30 * self.size)))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2d.setFont(font)
        self.pushButton_2d.setObjectName("pushButton_2d")

        self.widget_2d = QtWidgets.QWidget(self.frame_2d)
        self.widget_2d.setGeometry(QtCore.QRect(int(0 * self.size), int(40 * self.size), int(320 * self.size), int(280 * self.size)))
        self.widget_2d.setObjectName("widget_2d")

        self.verticalLayout_2d = QtWidgets.QVBoxLayout(self.widget_2d)
        self.verticalLayout_2d.setGeometry(QtCore.QRect(int(0 * self.size), int(0 * self.size), int(320 * self.size), int(280 * self.size)))
        self.verticalLayout_2d.setObjectName("verticalLayout_2d")


        # 3D
        self.frame_3d = QtWidgets.QFrame(self.centralwidget)
        self.frame_3d.setGeometry(QtCore.QRect(int(640 * self.size), int(0 * self.size), int(320 * self.size), int(320 * self.size)))
        self.frame_3d.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3d.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3d.setObjectName("frame_3d")

        self.pushButton_3d = QtWidgets.QPushButton(self.frame_3d)
        self.pushButton_3d.setGeometry(QtCore.QRect(int(110 * self.size), int(5 * self.size), int(100 * self.size), int(30 * self.size)))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3d.setFont(font)
        self.pushButton_3d.setObjectName("pushButton_3d")

        self.widget_3d = QtWidgets.QWidget(self.frame_3d)
        self.widget_3d.setGeometry(QtCore.QRect(int(0 * self.size), int(40 * self.size), int(320 * self.size), int(280 * self.size)))
        self.widget_3d.setObjectName("widget_3d")

        self.verticalLayout_3d = QtWidgets.QVBoxLayout(self.widget_3d)
        self.verticalLayout_3d.setGeometry(QtCore.QRect(int(0 * self.size), int(0 * self.size), int(320 * self.size), int(280 * self.size)))
        self.verticalLayout_3d.setObjectName("verticalLayout_3d")


        # Control
        self.frame_control = QtWidgets.QFrame(self.centralwidget)
        self.frame_control.setGeometry(QtCore.QRect(int(0 * self.size), int(320 * self.size), int(960 * self.size), int(100 * self.size)))
        self.frame_control.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_control.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_control.setObjectName("frame_control")

        self.pushButton_ready = QtWidgets.QPushButton(self.frame_control)
        self.pushButton_ready.setGeometry(QtCore.QRect(int(20 * self.size), int(10 * self.size), int(280 * self.size), int(60 * self.size)))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_ready.setFont(font)
        self.pushButton_ready.setObjectName("pushButton_ready")

        self.pushButton_start = QtWidgets.QPushButton(self.frame_control)
        self.pushButton_start.setGeometry(QtCore.QRect(int(340 * self.size), int(10 * self.size), int(600 * self.size), int(80 * self.size)))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
        self.pushButton_start.setEnabled(False)
        self.pushButton_start.clicked.connect(self.StartStart)
        self.Block_points = np.arange(0)

        self.progressBar_ready = QtWidgets.QProgressBar(self.frame_control)
        self.progressBar_ready.setGeometry(QtCore.QRect(int(20 * self.size), int(70 * self.size), int(280 * self.size), int(20 * self.size)))
        self.progressBar_ready.setProperty("value", 24)
        self.progressBar_ready.setObjectName("progressBar_ready")
        self.timer = QBasicTimer() # python QBasicTimer can only be used with threads started with Qthread
        #self.step = 0
        self.pushButton_ready.clicked.connect(self.ActionReady)


        # Result
        self.label_result = QtWidgets.QLabel(self.centralwidget)
        self.label_result.setGeometry(QtCore.QRect(int(0 * self.size), int(420 * self.size), int(960 * self.size), int(70 * self.size)))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_result.setFont(font)
        self.label_result.setStyleSheet("color: red;" "background-color: #000000")
        self.label_result.setAcceptDrops(False)
        self.label_result.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_result.setAutoFillBackground(False)
        self.label_result.setAlignment(QtCore.Qt.AlignCenter)
        self.label_result.setWordWrap(False)
        self.label_result.setObjectName("label_result")

        self.ActionCamera()


        MainWindow.setCentralWidget(self.centralwidget)


        # Menubar
        settingsAction = QAction(QIcon('C:/Users/RKH/PycharmProjects/FlyCV/Resources/settings.png'), 'Settings', self)
        settingsAction.setShortcut('Ctrl+E')
        settingsAction.setStatusTip('Edit settings')
        settingsAction.triggered.connect(self.ActionSettings)

        exitAction = QAction(QIcon('C:/Users/RKH/PycharmProjects/FlyCV/Resources/exit.png'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit program')
        exitAction.triggered.connect(qApp.quit)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        filemenu = self.menubar.addMenu('&File')
        filemenu.addAction(settingsAction)
        filemenu.addAction(exitAction)
        self.menubar.setGeometry(QtCore.QRect(int(0 * self.size), int(0 * self.size), int(960 * self.size), int(26 * self.size)))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)


        # Statusbar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.ActionStatusbar()


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def ActionSettings(self):
        self.t_Setting = SettingThread(self)
        self.t_Setting.start()

    def ActionCamera(self):
        t_Camera = CameraThread(self)
        t_Camera.start()

    def timerEvent(self, e): # Event handler that can be reimplemented in a subclass to receive timer events for the object.
        if e.timerId() != self.timer:   # https://wikidocs.net/36589
                                        # ???timerEvent 함수 하나로 여러개의 타이머에 대한 제어를 어떻게 하는지 아직 모르겠다.
            if self.step >= 100:
                self.ard.digital[self.ID_LIGHT].write(0)
                self.timer.stop()
                self.pushButton_ready.setText('Ready')
                self.pushButton_start.setEnabled(True)
                return
            self.step += 1
            self.progressBar_ready.setValue(self.step)
        # elif e.timerId() != self.timer_start:
        #     if self.step_start == 0:
        #         self.timer_start.stop()
        #         self.pushButton_start.setText('Start')
        #         self.pushButton_ready.setEnabled(True)
        #         return
        #     self.pushButton_start.setValue(self.step_start)
        #     self.step_start -= 1

    def ActionReady(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pushButton_ready.setText('Ready')
        else:
            self.timer.start(30, self)  # if changed, progressbar time will be changed (milsec)
            self.pushButton_ready.setText('Stop')
            self.pushButton_start.setEnabled(False) # https://pythonpyqt.com/qtimer/
            self.step = 0
            self.ard.digital[self.ID_LIGHT].write(1)

    def StartStart(self): # https://blog.naver.com/PostView.naver?blogId=jeong001116&logNo=222284724196&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView
        self.timer_start = QTimer(self)
        self.step_start = 5
        self.timer_start.start(500)
        self.pushButton_start.setEnabled(False)
        self.pushButton_ready.setEnabled(False)
        self.timer_start.timeout.connect(self.ActionStart)

    def ActionStart(self):
        if self.step_start == 0:
            self.timer_start.stop()
            self.pushButton_start.setText('Start')
            self.progressBar_ready.setValue(0)
            self.step_start = 5

            self.Block_points = np.arange(100)
            self.pushButton_ready.setText("Analyzing..")
            self.pushButton_ready.setEnabled(False)
            self.pushButton_start.setText("Analyzing..")
        else:
            self.pushButton_start.setText(str(self.step_start))
            self.step_start -= 1

    def ActionStatusbar(self):
        t_Statusbar = StatusbarThread(self)
        t_Statusbar.start()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "BRL_UI_demo"))
        self.pushButton_camera.setText(_translate("MainWindow", "Camera"))
        self.label_result.setText(_translate("MainWindow", "Warning (Suspicious)"))
        self.pushButton_2d.setText(_translate("MainWindow", "2D"))
        self.pushButton_3d.setText(_translate("MainWindow", "3D"))
        self.pushButton_ready.setText(_translate("MainWindow", "Ready"))
        self.pushButton_start.setText(_translate("MainWindow", "Start"))


class SettingThread(QThread):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        cv2.namedWindow("TrackBars")
        cv2.resizeWindow("TrackBars", 720, 360)
        cv2.createTrackbar("topleft_x", "TrackBars", self.parent.topleft_x, 1384, self.changeThreshold)
        cv2.createTrackbar("topleft_y", "TrackBars", self.parent.topleft_y, 1032, self.changeThreshold)
        cv2.createTrackbar("bottomright_x", "TrackBars", self.parent.bottomright_x, 1384, self.changeThreshold)
        cv2.createTrackbar("bottomright_y", "TrackBars", self.parent.bottomright_y, 1032, self.changeThreshold)
        cv2.createTrackbar("min_area", "TrackBars", self.parent.min_area, 100, self.changeThreshold)
        cv2.createTrackbar("max_area", "TrackBars", self.parent.max_area, 500, self.changeThreshold)
        #cv2.createButton("Initialization", self.initialization) # 초기화 버튼 만들고 싶다..

    def changeThreshold(self, a):
        self.parent.topleft_x = cv2.getTrackbarPos("topleft_x", "TrackBars")
        self.parent.topleft_y = cv2.getTrackbarPos("topleft_y", "TrackBars")
        self.parent.bottomright_x = cv2.getTrackbarPos("bottomright_x", "TrackBars")
        self.parent.bottomright_y = cv2.getTrackbarPos("bottomright_y", "TrackBars")
        self.parent.min_area = cv2.getTrackbarPos("min_area", "TrackBars")
        self.parent.max_area = cv2.getTrackbarPos("max_area", "TrackBars")

    def initialization(self, a):
        self.parent.topleft_x = 250
        self.parent.topleft_y = 58
        self.parent.bottomright_x = 1344
        self.parent.bottomright_y = 950
        self.parent.min_area = 18
        self.parent.max_area = 200

    def run(self):
        print("Setting menu is triggered")



class CameraThread(QThread):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.lane_names = self.parent.lane_names

        ### setting for 2D plot
        self.fig = plt.Figure()
        ## set the plot background color to the window background color
        color = QtGui.QPalette().window().color()
        self.fig.set_facecolor((color.redF(), color.greenF(), color.blueF()))
        #plt.subplots_adjust(wspace=0, hspace=0)    # UI 외에 plt창에 관련된 것이다.(not recommended)
        self.canvas = FigureCanvas(self.fig)
        self.parent.verticalLayout_2d.addWidget(self.canvas)
        self.create_position_plots_2d(6)

        ### setting for 3D plot
        self.fig_3d = plt.Figure()
        color_3d = QtGui.QPalette().window().color()
        self.fig_3d.set_facecolor((color_3d.redF(), color_3d.greenF(), color_3d.blueF()))
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.parent.verticalLayout_3d.addWidget(self.canvas_3d)
        self.create_xvel_plots_3d(1)

        self.pushButton_ready = self.parent.pushButton_ready
        self.pushButton_start = self.parent.pushButton_start

    def create_position_plots_2d(self, num_plots):
        self.fig.clear()
        self.axes = [self.fig.add_subplot(num_plots, 1, n + 1) for n in range(num_plots)]
        for ax in self.axes[:-1]:
            ax.set_xticks([])
            ax.set_xlim((-0.5, 9.5))
            ax.set_ylim((-5, 20))
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_position(('outward', 5))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #ax.sharex(self.axes[-1])
        self.axes[-1].set_xlim((-0.5, 9.5))
        self.axes[-1].set_xticks((np.linspace(0, 9, 10)))
        self.axes[-1].set_ylim((-5, 20))
        self.axes[-1].spines['bottom'].set_position(('outward', 5))
        self.axes[-1].spines['left'].set_position(('outward', 5))
        self.axes[-1].spines['right'].set_visible(False)
        self.axes[-1].spines['top'].set_visible(False)
        #self.fig.tight_layout()
        self.canvas.draw()

    def create_xvel_plots_2d(self, num_plots):
        self.fig.clear()
        self.axes = [self.fig.add_subplot(num_plots, 1, n + 1) for n in range(num_plots)]
        x = np.linspace(1, 6, 6)
        for ax in self.axes:
            ax.set_xlim((0.5, 6.5))
            ax.set_xticks(x)
            ax.set_xticklabels(self.lane_names) # https://stackoverflow.com/questions/8384120/equivalent-function-for-xticks-for-an-axessubplot-object
            ax.set_ylim((-20, 60))
            ax.spines['bottom'].set_position(('outward', 5))
            ax.spines['left'].set_position(('outward', 5))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        self.canvas.draw()

    def create_xvel_plots_3d(self, num_plots):
        self.fig_3d.clear()
        self.axes_3d = [self.fig_3d.add_subplot(num_plots, 1, n + 1, projection='3d') for n in range(num_plots)] # https://github.com/jeev20/3D-static-and-dynamic-plots-with-Matplotlib-and-Pyqt/blob/master/3DMatplotlib_Pyqt.py
        for ax in self.axes_3d:
            ax.set_xlim((-20, 60))
            ax.set_ylim((-20, 60))
            ax.set_zlim((-20, 60))
            ax.set_xlabel(self.lane_names[1])
            ax.set_ylabel(self.lane_names[3])
            ax.set_zlabel(self.lane_names[5])
        self.canvas_3d.draw()

    def run(self):
        '''
        dur_cleaning = 26  # ?
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
        '''

        n_inds = 400
        colours = []
        for cnum in range(n_inds):
            colours.append((randint(0, 255), randint(0, 255), randint(0, 255)))

        scaling = 1
        block_size = int(25 * scaling / 2) * 2 + 1
        offset = int(25 * scaling)

        mot = True
        meas_last = list(np.zeros((n_inds, 2)))
        self.meas_now = list(np.zeros((n_inds, 2)))
        self.df = []
        Points = []

        # mask area
        # x
        topleft_x = int(240 * scaling)
        # y
        topleft_y = int(59 * scaling)
        # w
        bottomright_x = int(1350 * scaling)
        # h
        bottomright_y = int(958 * scaling)

        self.xnum = 10
        self.ynum = 6
        self.xborders = list(np.linspace(topleft_x, bottomright_x, (self.xnum + 1)))
        self.yborders = list(np.linspace(topleft_y, bottomright_y, (self.ynum + 1)))
        print('xborders', self.xborders)
        print('yborders', self.yborders)

        cap = cv2.VideoCapture('C:/Users/RKH/PycharmProjects/FlyCV/output/20220120/Day/Cartridge1_W1118_W1118_W1118_W1118_W1118_Or7a_D_DPG_1-heptanol_2-pentanol_1AA_BA/t20220120_182854.mp4')
        if not cap.isOpened():
            raise Exception("Could not open the video device")
        FPS = cap.get(cv2.CAP_PROP_FPS)
        print(FPS)
        self.frame_num = 0
        while cap.isOpened():
            topleft_x = self.parent.topleft_x
            topleft_y = self.parent.topleft_y
            bottomright_x = self.parent.bottomright_x
            bottomright_y = self.parent.bottomright_y
            min_area = self.parent.min_area
            max_area = self.parent.max_area

            ret, frame = cap.read()
            if ret:
                bg = frame.copy()
                mask = np.zeros((frame.shape[0], frame.shape[1]))
                cv2.rectangle(mask, (topleft_x, topleft_y), (bottomright_x, bottomright_y), 255, cv2.FILLED)
                cv2.rectangle(mask, (bottomright_x - 9, bottomright_y - 150), (bottomright_x, bottomright_y), 0, -1)
                frame[mask == 0] = 0
                thresh = tr.colour_to_thresh(frame, block_size, offset)
                final, contours, meas_last, self.meas_now = tr.detect_and_draw_contours(frame, thresh, meas_last, self.meas_now, min_area, max_area)

                meas_last = list(meas_last)
                meas_now = list(self.meas_now)
                meas_now.sort(key=lambda x: x[1])
                meas_now.sort(key=lambda x: x[0])

                row_ind, col_ind = tr.hungarian_algorithm(meas_last, self.meas_now)
                final, self.meas_now, self.df = tr.reorder_and_draw(final, colours, n_inds, col_ind, self.meas_now, self.df, mot)
                final[mask == 0] = bg[mask == 0]

                if self.frame_num in self.parent.Block_points:
                    if self.frame_num == 0:
                        self.create_position_plots_2d(6)
                        self.df = []
                    self.count = 0
                    self.newPoints = []
                    if (self.frame_num % 20) == 0:
                        self.populations_new = []
                    for i in range(len(self.meas_now)):
                        self.x = [int(x) for x in self.meas_now[i]][0]
                        self.y = [int(x) for x in self.meas_now[i]][1]
                        if self.x != 0 and self.y != 0:
                            self.df.append([self.frame_num, self.x, self.y, self.count])
                            self.newPoints.append([self.x, self.y, self.count])
                            if (self.frame_num % 20) == 0:
                                self.populations_new.append([self.x, self.y])
                        self.count += 1
                    if (self.frame_num % 20) == 0:
                        self.t_pre_2D3D = Thread_pre_2D3D(self)
                        self.t_pre_2D3D.start()
                    if len(self.newPoints) != 0:
                        for newP in self.newPoints:
                            Points.append(newP)
                    if (len(Points) != 0):
                        self.drawOnCanvas(final, Points, colours)
                    self.frame_num += 1
                    if self.frame_num == 100:
                        self.frame_num = 0
                        self.parent.Block_points = np.arange(0)
                        Points = []
                        self.create_xvel_plots_2d(1)
                        self.create_xvel_plots_3d(1) # 지우면 누적 plotting 가능.
                        self.t_2D3D = Thread_2D3D(self)
                        self.t_2D3D.start()

                RGBimg = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
                h, w, ch = RGBimg.shape
                Qimg = QImage(RGBimg.data, w, h, QImage.Format_RGB888)
                pixmap = QPixmap(Qimg)
                p = pixmap.scaled(int(300 * self.parent.size), int(270 * self.parent.size), Qt.KeepAspectRatio)
                self.parent.label_camera.setPixmap(p)

                if cv2.waitKey(10) and 0xff == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def drawOnCanvas(self, final, myPoints, myColorValues):
        for point in myPoints:
            cv2.circle(final, (point[0], point[1]), 1, myColorValues[point[2]], cv2.FILLED)


class Thread_pre_2D3D(QThread): # showing realtime positions
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.xborders = self.parent.xborders
        self.yborders = self.parent.yborders
        self.populations_new = pd.DataFrame(self.parent.populations_new, columns=['pos_x', 'pos_y'])
        self.PN = np.zeros((self.parent.ynum, self.parent.xnum))

        self.axes = self.parent.axes

    def run(self):
        for j in range(self.parent.ynum):
            time.sleep(0.01)
            for k in range(self.parent.xnum):
                self.PN[j][k] = self.populations_new[(self.populations_new['pos_x'] >= self.xborders[k]) & (self.populations_new['pos_x'] < self.xborders[k + 1]) & (self.populations_new['pos_y'] >= self.yborders[j]) & (self.populations_new['pos_y'] < self.yborders[j + 1])].count()[0]
        # print('populations_new', self.PN)
        for i in range(6):
            self.axes[i].plot(self.PN[i], color='m', alpha=((0.1 * (self.parent.frame_num // 20)) + 0.1))
        self.parent.canvas.draw()


class Thread_2D3D(QThread):   # showing velocity on each odor
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.yborders = self.parent.yborders
        self.df = self.parent.df
        self.lane_names = self.parent.lane_names

        self.axes = self.parent.axes
        self.axes_3d = self.parent.axes_3d

    def run(self):
        self.df = pd.DataFrame(self.df, columns=['frame', 'pos_x', 'pos_y', 'fly_id'])
        max_frame_index = self.df.at[self.df.index[-1], 'frame']
        max_fly_index = max(self.df['fly_id'])
        print(max_frame_index, ' ', max_fly_index)
        xpos = np.empty((max_fly_index, max_frame_index))
        xpos[:] = np.nan
        ypos = np.empty((max_fly_index, max_frame_index))
        ypos[:] = np.nan
        for i in range(len(self.df)):
            fly_idx = int(self.df.iloc[i]['fly_id'])
            frame_idx = int(self.df.iloc[i]['frame'])
            xpos[fly_idx - 1][frame_idx - 1] = self.df.iloc[i]['pos_x']
            ypos[fly_idx - 1][frame_idx - 1] = self.df.iloc[i]['pos_y']
            if (abs(xpos[fly_idx - 1][frame_idx - 1] - xpos[fly_idx - 1][max(0, frame_idx - 2)]) > 25) or (abs(ypos[fly_idx - 1][frame_idx - 1] - ypos[fly_idx - 1][max(0, frame_idx - 2)]) > 25):
                xpos[fly_idx - 1][frame_idx - 2] = np.nan
                ypos[fly_idx - 1][frame_idx - 2] = np.nan
        xpos_diff = np.diff(xpos)
        lane_num = np.zeros(ypos.shape)
        for i in range(len(self.lane_names)):
            lane_num[(ypos < self.yborders[i + 1]) & (ypos >= self.yborders[i])] = i + 1
        lane_num = lane_num[:, 1:]
        xvel = np.empty((len(self.lane_names), max_frame_index - 1))
        xvel[:] = np.nan
        for i in range(len(self.lane_names)):
            temp = np.empty(xpos_diff.shape)
            temp[:] = xpos_diff
            temp[lane_num != (i + 1)] = np.nan
            xvel[i, :] = np.nanmean(temp, axis=0)
        response_by_xvel = np.zeros(len(self.lane_names))
        for i in range(len(self.lane_names)):
            response_by_xvel[i] = np.nanmean(xvel[i, :])
        self.axes[0].bar(np.linspace(1, 6, 6), (-10) * response_by_xvel)
        self.axes[0].set_xticks(np.linspace(1, 6, 6))
        self.axes[0].set_xticklabels(self.lane_names)
        self.parent.canvas.draw()
        print('response_by_xvel', response_by_xvel)

        self.axes_3d[0].scatter((-10) * response_by_xvel[1], (-10) * response_by_xvel[3], (-10) * response_by_xvel[5])
        self.parent.canvas_3d.draw()

        self.parent.pushButton_ready.setEnabled(True)
        self.parent.pushButton_ready.setText("Ready")
        self.parent.pushButton_start.setText("Start")


class StatusbarThread(QThread):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def run(self):
        self.now = QDateTime.currentDateTime()
        self.parent.statusbar.showMessage("{}    Temperature: {} 'C    Humidity: {} %".format(self.now.toString(Qt.DefaultLocaleShortDate), 30, 20))
        timer = Timer(1, self.run)
        timer.start()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = BRL_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())