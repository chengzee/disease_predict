from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QIcon, QPixmap
from V8 import Ui_MainWindow
import sys
import pyqtgraph as pg
import time
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from function_Encoder_Decoder import Encoder, Decoder

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
        # graphicsViewII
        self.apply_predict_model_for_thermal_plane()
        self.show_predicted_thermal_plane()
        # self.getFixFointData()
        self.timer = QtCore.QTimer()

        # 每 2 分鐘刷新
        self.timer.setInterval(120*1000)
        # self.timer.timeout.connect(self.changeFollowBedDisplay, self.getFixFointData)
        self.timer.timeout.connect(self.changeFollowBedDisplay)
        self.timer.start()

        # self.show_img()

    def setUp(self):
        # MainWindow Title
        self.setWindowTitle('蘭花微氣候監測系統')
        # # label1
        # self.ui.label1.setText('第六區監測')
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

        # label
        self.ui.label.setText('病害嚴重程度預估：')

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
        # Set a custom color map 
        colors = [ 
            (255, 255, 0), 
            (50, 205, 50), 
            (0, 0, 198)
        ] 
  
        # color map 
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3), color=colors) 
  
        # setting color map to the image view 
        self.ui.graphicsViewI.setColorMap(cmap)
        self.ui.graphicsViewI.setImage(humid_interpolation.T)
        # self.ui.graphicsViewI.histogram.hide()

    def apply_predict_model_for_thermal_plane(self):
        self.fs_temp = [[], [], [], [], [], [], [], [], [], [], [], []]
        self.fs_humid = [[], [], [], [], [], [], [], [], [], [], [], []]
        self.fs_light = [[], [], [], [], [], [], [], [], [], [], [], []]
        self.fs_timestamp = [[], [], [], [], [], [], [], [], [], [], [], []]
        self.predictions= [[], [], [], [], [], [], [], [], [], [], [], []]
        self._lookback = 288
        self._source_dim = 3
        self._predict_dim = 1
        self._delay = 24
        self.data_min = np.array((47.14, 24.66, 0))
        self.data_max = np.array((99.9, 36.54, 2286.832))
        self.neuron = 1024
        self.A = 0
        self.BATCH_SIZE = 128
        self.width, self.length = (3, 4)
        self.humid_thermal_plane_24 = np.zeros((self.width, self.length))
        self.temp_thermal_plane_24 = np.zeros((self.width, self.length))
        # build fixed sensors dataset
        for fs in range(12):
            dataUrl = url + "/" + str(fs+1)
            print(dataUrl)
            getjsons = requests.get(dataUrl).json()
            # print(len(getjsons["data"]))
            # 取欲預測的資料長度
            for n in range(len(getjsons['data'][-self._lookback-1:-1])):
                print(type(fs))
                print(type(n-self._lookback-1))
                self.fs_temp[fs].append(getjsons['data'][n-self._lookback-1]['temperature'])
                self.fs_humid[fs].append(getjsons['data'][n-self._lookback-1]['wetness'])
                self.fs_light[fs].append(getjsons['data'][n-self._lookback-1]['par']/55)
                self.fs_timestamp[fs].append(getjsons['data'][n-self._lookback-1]['time'])
            temp_arr = np.array(self.fs_temp[fs]).reshape(-1, 1)
            humid_arr = np.array(self.fs_humid[fs]).reshape(-1, 1)
            light_arr = np.array(self.fs_light[fs]).reshape(-1, 1)
            timestamp_arr = np.array(self.fs_timestamp[fs]).reshape(-1, 1)
            # 將各特徵結合至新矩陣後，進行資料前處理，才套用模型
            rawdata_array = np.append(np.append(np.append(timestamp_arr, humid_arr, axis=1), temp_arr, axis=1), light_arr, axis=1)
            # print(rawdata_array.shape)
            print(rawdata_array.shape)
            paddeddata_array = rawdata_array
            total_LossNumber = 0
            # paddeddata 針對缺漏資料點以線性插植方式補上
            for n in range(len(rawdata_array)):
                if n > 0:
                    # 計算時間差 (轉換為 unix timestamp)
                    fronttime = rawdata_array[n-1, 0]
                    latertime = rawdata_array[n, 0]
                    diff_sec = (latertime-fronttime)/1000
                    diff_step = diff_sec/300 
                    print("diff_step:{}".format(diff_step))
                    if diff_step >= 2:
                        LossNumber = int(diff_step)-1
                        time_interval = diff_sec/(LossNumber+1)
                        LossValue = (rawdata_array[n:n+1, 1:4]-rawdata_array[n-1:n, 1:4])/(LossNumber+1)
                        # 如果二小時的關機後再重啟...
                        if diff_step > 24:
                            padding_array = np.zeros((LossNumber, 4))
                        if diff_step <= 24:
                            padding_array = np.ones((LossNumber, 4))
                        for v in range(LossNumber):
                            padding_array[v:v+1, 0:1] = rawdata_array[n-1, 0] + time_interval*(v+1)
                            padding_array[v:v+1, 1:4] = rawdata_array[n-1:n, 1:4] + LossValue*(v+1)
                        paddeddata_array = np.vstack((np.vstack((paddeddata_array[:n+total_LossNumber], padding_array)), paddeddata_array[n+total_LossNumber:]))    
                        total_LossNumber += LossNumber
            # 最大最小值正規化 Normalization (min-Max normalization)
            paddeddata_array_norm = (paddeddata_array[:, 1:4]-self.data_min)/(self.data_max-self.data_min)
            # print(paddeddata_array_norm)
            # Reshape為可應用於模型之矩陣
            paddeddata_array_norm = paddeddata_array_norm.reshape((1, -1, 3))
            # print(paddeddata_array_norm.shape)
            # 模型建立與權重載入
            # model functional
            input_seq = tf.keras.Input(shape=(self._lookback, self._source_dim))
            encoder_stack_h, enc_last_h = Encoder(self.neuron, self.A, self._lookback, self._source_dim)(input_seq)
            decoder_input = tf.keras.layers.RepeatVector(self._delay)(enc_last_h)
            decoder_stack_h = Decoder(self.neuron, self.A)(decoder_input)
            # attention layer
            attention = tf.keras.layers.dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
            attention = tf.keras.layers.Activation('softmax')(attention)
            # print(attention)
            context = tf.keras.layers.dot([attention, encoder_stack_h], axes=[2, 1])
            # print(context)
            decoder_combine_context = tf.keras.layers.concatenate([context, decoder_stack_h])
            # print(decoder_combine_context)
            out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self._predict_dim))(decoder_combine_context)
            model = tf.keras.models.Model(inputs=input_seq, outputs=out)

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            model.summary()
            # checkpoint
            # filepath = save_file_path + "/weights288.best.h5"
            filepath = "weights288.best.h5"
            model.load_weights(filepath)
            self.predictions[fs] = model.predict(paddeddata_array_norm[:, -self._lookback:, 0:3], verbose=1, batch_size=self.BATCH_SIZE)
            print("self.predictions[fs].shape:{}".format(self.predictions[fs].shape))
        
    def show_predicted_thermal_plane(self):
        # for pred in range(self._delay):

        for fs in range(12):
            # self.humid_thermal_plane_24[fs%3, int(fs/3)] = self.predictions[fs][0:1, -(pred+1), 0:1]
            self.humid_thermal_plane_24[fs%3, int(fs/3)] = self.predictions[fs][0:1, -1, 0:1]
        
        # # update thermal plane in GUI----------------------------------------------------------------------
        # 在opencv中是(x, y)
        # dst_shape = (39, 28)
        # dst_shape = (2600, 2800)
        dst_shape = (3900, 2800)
        # dst_shape = (2800, 3900)
        self.humid_interpolation = cv2.resize(self.humid_thermal_plane_24, dst_shape)
        self.temp_interpolation = cv2.resize(self.temp_thermal_plane_24, dst_shape)
        print(self.humid_interpolation.shape)
        print(self.temp_interpolation.shape)
        # Set a custom color map 
        colors = [ 
            (255, 255, 0), 
            (50, 205, 50), 
            (0, 0, 198)
        ] 
        # color map 
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 3), color=colors) 
        # setting color map to the image view 
        self.ui.graphicsViewII.setColorMap(cmap)
        self.ui.graphicsViewII.setImage(self.humid_interpolation.T)

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
        for n in range(len(getjsons['data'][-150:])):
            self.temp.append(getjsons['data'][n-150]['temperature'])
            self.humid.append(getjsons['data'][n-150]['wetness'])
            self.light.append(getjsons['data'][n-150]['par']/55)
            self.timestamp.append(getjsons['data'][n-150]['time'])
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
        
    def show_Features(self):
        for n in range(len(self.gV)):
            self.gV[n].clear()
            self.gV[n].plot(self.timestamp, self.paras[n], pen=pg.mkPen(color=self.colors[n], width=3))
            # self.ui.graphicsView_2.setXRange(self.time-24*60*60*3, self.time)
            pg.QtGui.QApplication.processEvents()


if __name__ == '__main__':
     app = QtWidgets.QApplication([])
     window = MainWindow()
     window.show()
     sys.exit(app.exec_())
