from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QIcon, QPixmap
from V6 import Ui_MainWindow
import sys
import pyqtgraph as pg
import time
import cv2
import pandas as pd
import numpy as np

import requests
url = "https://monitor.icmems.ml/api/getDatas"
# print(getjsons.json())
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setUp()
        # ComboBox1
        self.ui.comboBox1.addItems(self.section)
        # self.ui.comboBox1.currentIndexChanged.connect(self.display)
        self.ui.comboBox1.currentIndexChanged
        # ComboBox2
        self.ui.comboBox2.addItems(self.sensors)
        self.ui.comboBox2.currentIndexChanged.connect(self.changeFollowBedDisplay)
        # # checkBox1
        # self.ui.checkBox1.stateChanged.connect(self.show_Features)
        # # checkBox2
        # self.ui.checkBox2.stateChanged.connect(self.show_Features)
        # # checkBox3
        # self.ui.checkBox3.stateChanged.connect(self.show_Features)
        self.changeFollowBedDisplay()
        # graphicsViewI
        self.update_current_thermal_plane()
        # self.getFixFointData()
        self.timer = QtCore.QTimer()

        # 每 2 分鐘刷新
        self.timer.setInterval(120*1000)
        # self.timer.timeout.connect(self.changeFollowBedDisplay, self.getFixFointData)
        self.timer.timeout.connect(self.changeFollowBedDisplay)
        self.timer.start()
        self.show_img()

    def setUp(self):
        # MainWindow Title
        self.setWindowTitle('蘭花微氣候監測系統')
        # label1
        self.ui.label1.setText('第六區監測')
        # label2
        self.ui.label2.setText('隨床監測：')
        # groupBox
        self.ui.groupBox.setTitle('第六區')
        # groupBox_2
        self.ui.groupBox_2.setTitle('第六區預測')
        # # global checkBoxes
        # self.checkBoxes = [self.ui.checkBox1, self.ui.checkBox2, self.ui.checkBox3]
        # ComboBox1
        self.section = ['第六區']
        # ComboBox2
        self.sensors = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']
        # # checkBox1
        # self.ui.checkBox1.setText('溫度 (Temp)')
        # # checkBox2
        # self.ui.checkBox2.setText('濕度 (Humid)')
        # # checkBox3
        # self.ui.checkBox3.setText('光照量 (micromol)')
        # time
        self.time = time.time()
        # label-'voltage_meter'
        self.ui.voltage_meter.setText('電池更換警示')
        # label-'update_time'
        self.ui.update_time.setText('上次更新時間:')
        # color setting 
        self.colors = ["#D3D4", "#4B88A2", "#BB0A21"]
        labelStyle = {'color': '#000000', 'font-size': '8pt'}
        
        # graphicsView_2, 3, 4 
        self.gV = [self.ui.graphicsView_2, self.ui.graphicsView_3, self.ui.graphicsView_4]
        # graphicsView_2 setting
        # self.para_names = ['temperature', 'humidity', 'light']
        self.ui.graphicsView_2.setLabel('left', "Temperature", units='Celsius', **labelStyle)
        self.ui.graphicsView_2.setBackground('w')
        # graphicsView_3 setting
        self.ui.graphicsView_3.setLabel('left', "Relative Humidity", units='%', **labelStyle)
        self.ui.graphicsView_3.setBackground('w')
        # graphicsView_4 setting
        # Photosynthetically Active Radiation
        self.ui.graphicsView_4.setLabel('left', "PAR", units='micro mol/sec m^2', **labelStyle)
        self.ui.graphicsView_4.setBackground('w')
        # graphicsView setting
        self.fixedsensors = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

    def update_current_thermal_plane(self):
        width, length = (3, 4)
        # width, length = (3, 3)
        humid= np.zeros((width, length))
        temp = np.zeros((width, length))
        for w in range(width):
            for l in range(length):
                try:
                    # rawdata = pd.read_csv("sensorNode{}.csv".format(3*l+w+1)) # 讀資料進來
                    # rawdata_array = np.array(rawdata) # 轉為矩陣
                    # 讀資料進來
                    fixedPoint_dataUrl = url+"/"+self.fixedsensors[3*l+w]
                    print(fixedPoint_dataUrl)
                    fP_getjson = requests.get(fixedPoint_dataUrl).json()
                    humid[w, l] = fP_getjson['data'][-1]['wetness']
                    temp[w, l] = fP_getjson['data'][-1]['temperature']
                    # print(humid)
                    # print(temp)
                except FileNotFoundError:
                    print("FileNotFoundError")
                    # humid[pose] = 0
                    # temp[pose] = 0
                except :
                    print("Unexpected Error")
        # 在opencv中是(x, y)
        # dst_shape = (39, 28)
        # dst_shape = (2600, 2800)
        dst_shape = (3900, 2800)
        # dst_shape = (2800, 3900)
        humid_interpolation = cv2.resize(humid, dst_shape)
        temp_interpolation = cv2.resize(temp, dst_shape)
        print(humid_interpolation.shape)
        print(temp_interpolation.shape)
        # self.ui.graphicsViewI.show()
        self.ui.graphicsViewI.setImage(humid_interpolation.T)

    def show_img(self):
        # pix = QPixmap(image_path)
        # label_image = QLabel()
        # label_image.setPixmap(QPixmap(image_path))
        
        # # Create widget
        # self.ui.graphicsViewII = QLabel(self)
        # pixmap = QPixmap("C:/Users/HUANG/Desktop/luongatt/5th_512neurons_1ENC-DEC_288-72_1.png")
        # self.ui.graphicsViewII.setPixmap(pixmap)
        # # self.resize(pixmap.width(),pixmap.height())
        
        # self.ui.graphicsViewII.show()

        
        scene = QtGui.QGraphicsScene()
        img_path = "C:/Users/HUANG/Desktop/luongatt/5th_512neurons_1ENC-DEC_288-72_1.png"
        #scene.setSceneRect(-600,-600, 600,600)
        scene.setSceneRect(-600, -600, 1200, 1200)

        pic = QtGui.QPixmap(img_path)
        scene.addItem(QtGui.QGraphicsPixmapItem(pic))
        # view = self.gv
        self.ui.graphicsViewII.setScene(scene)
        self.ui.graphicsViewII.setRenderHint(QtGui.QPainter.Antialiasing)
        self.ui.graphicsViewII.show()



    def changeFollowBedDisplay(self):
        followBed = self.ui.comboBox2.currentText()
        # print(followBed[1])
        dataUrl = url+"/9"+followBed[1]
        getjsons = requests.get(dataUrl).json()
        # print(dataUrl)
        
        self.temp = []
        self.humid = []
        self.light = []
        self.timestamp = []
        for n in range(len(getjsons['data'][-288*1:])):
            self.temp.append(getjsons['data'][n-288*1]['temperature'])
            self.humid.append(getjsons['data'][n-288*1]['wetness'])
            self.light.append(getjsons['data'][n-288*1]['par'])
            self.timestamp.append(getjsons['data'][n-288*1]['time'])
        print(len(self.timestamp))
        print(len(self.temp))
        print(len(self.humid))
        print(len(self.light))
        print(self.time)
        self.paras = [self.temp, self.humid, self.light]
        self.show_Features()
        # Time~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.the_last_unixTC = getjsons['data'][-1]['time']/1000
        self.struct_time = time.localtime(self.the_last_unixTC)
        # print(type(self.the_last_unixTC))
        self.date_form = time.strftime("%Y-%m-%d %H:%M:%S", self.struct_time)
        print(type(self.date_form))
        self.ui.update_time.setText('上次更新時間:{}'.format(self.date_form))
        # Time~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # def show_Features(self):
    #     self.ui.graphicsView_2.clear()
    #     for n in range(len(self.checkBoxes)):
    #         if self.checkBoxes[n].isChecked():
    #             # self.checkBoxes[n].setBackground(self.colors[n])
    #             print("add {} feature".format(self.para_names[n]))
    #             self.ui.graphicsView_2.plot(self.timestamp, self.paras[n], pen=pg.mkPen(color=self.colors[n], width=3))
    #             # self.ui.graphicsView_2.setXRange(self.time-24*60*60*3, self.time)
    #             pg.QtGui.QApplication.processEvents()
    def show_Features(self):
        for n in range(len(self.gV)):
            self.gV[n].clear()
            self.gV[n].plot(self.timestamp, self.paras[n], pen=pg.mkPen(color=self.colors[n], width=3))
            # self.ui.graphicsView_2.setXRange(self.time-24*60*60*3, self.time)
            pg.QtGui.QApplication.processEvents()

    # def getFixFointData(self):
    #     self.all_latest_temp = []
    #     self.all_latest_humid = []
    #     self.all_latest_light = []
    #     self.ui.graphicsView.clear()
    #     for n in self.fixedsensors:
    #         fixdataURL = url+"/"+self.fixedsensors[n]
    #         fixdata_getjsons = requests.get(fixdataURL).json()
    #         self.all_latest_temp.append(fixdata_getjsons['data'][-1]['temperature'])
    #         self.all_latest_humid.append(fixdata_getjsons['data'][-1]['wetness'])
    #         self.all_latest_light.append(fixdata_getjsons['data'][-1]['par'])
        


if __name__ == '__main__':
     app = QtWidgets.QApplication([])
     window = MainWindow()
     window.show()
     sys.exit(app.exec_())
