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
# from MA_luongAtt_byCZ import * 

import requests
url = "https://monitor.icmems.ml/api/getDatas"

temp = [[], [], [], [], [], [], [], [], [], [], [], []]
humid = [[], [], [], [], [], [], [], [], [], [], [], []]
light = [[], [], [], [], [], [], [], [], [], [], [], []]
timestamp = [[], [], [], [], [], [], [], [], [], [], [], []]
# print(len(temp))
for fs in range(12):
    dataUrl = url + "/" + str(fs+1)
    print(dataUrl)
    getjsons = requests.get(dataUrl).json()
    # print(len(getjsons["data"]))
    for n in range(len(getjsons['data'][-500:])):
        temp[fs].append(getjsons['data'][n-500]['temperature'])
        humid[fs].append(getjsons['data'][n-500]['wetness'])
        light[fs].append(getjsons['data'][n-500]['par']/55)
        timestamp[fs].append(getjsons['data'][n-500]['time'])
    temp_arr = np.array(temp[fs]).reshape(-1, 1)
    humid_arr = np.array(humid[fs]).reshape(-1, 1)
    light_arr = np.array(light[fs]).reshape(-1, 1)
    timestamp_arr = np.array(timestamp[fs]).reshape(-1, 1)
    # 將各特徵結合至新矩陣後，進行資料前處理，才套用模型
    rawdata_array = np.append(np.append(np.append(timestamp_arr, humid_arr, axis=1), temp_arr, axis=1), light_arr, axis=1)
    for n in range(len(rawdata_array)):
        if n > 0:
            # 計算時間差 (轉換為 unix timestamp)
            fronttime = rawdata_array[n-1, 0]
            latertime = rawdata_array[n, 0]
            diff_sec = latertime-fronttime
            diff_step = diff_sec/300 
            print("diff_step:{}".format(diff_step))
            if diff_step >= 2:
                LossNumber = int(diff_step)-1
                time_interval = diff_sec/(LossNumber+1)
                LossValue = (rawdata_array[n:n+1, 1:4]-rawdata_array[n-1:n, 1:4])/(LossNumber+1)
                # 如果二小時的關機後再重啟，會另行註記，之後將可能會不採用有包含該範圍的數據
                if diff_step > 24:
                    padding_array = np.zeros((LossNumber, 5))
                if diff_step <= 24:
                    padding_array = np.ones((LossNumber, 5))
                for v in range(LossNumber):
                    padding_array[v:v+1, 0:1] = rawdata_array[n-1, 0] + time_interval*(v+1)
                    padding_array[v:v+1, 1:4] = rawdata_array[n-1:n, 1:4] + LossValue*(v+1)
                paddeddata_array = np.vstack((np.vstack((paddeddata_array[:n+total_LossNumber], padding_array)), paddeddata_array[n+total_LossNumber:]))    
                total_LossNumber += LossNumber
