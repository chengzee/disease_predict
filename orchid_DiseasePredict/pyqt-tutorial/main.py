from PyQt5 import QtWidgets, QtGui, QtCore, QtChart
from PyQt5.QtWidgets import QApplication, QMainWindow
from V3 import Ui_MainWindow
import sys
import pyqtgraph as pg
import time
import cv2
from datetime import datetime
import requests
url = "https://monitor.icmems.ml/api/getDatas"
# print(getjsons.json())

# timestamp = 1545730073
# dt_object = datetime.fromtimestamp(timestamp)

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
        # checkBox1
        self.ui.checkBox1.stateChanged.connect(self.show_Features)
        # checkBox2
        self.ui.checkBox2.stateChanged.connect(self.show_Features)
        # checkBox3
        self.ui.checkBox3.stateChanged.connect(self.show_Features)
        self.changeFollowBedDisplay()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(60*1000)
        self.timer.timeout.connect(self.changeFollowBedDisplay)
        self.timer.start()

    def setUp(self):
        # MainWindow Title
        self.setWindowTitle('蘭花微氣候監測系統')
        # label1
        self.ui.label1.setText('第六區監測')
        # label2
        self.ui.label2.setText('隨床監測：')
        # groupBox
        self.ui.groupBox.setTitle('第六區')
        # global checkBoxes
        self.checkBoxes = [self.ui.checkBox1, self.ui.checkBox2, self.ui.checkBox3]
        # ComboBox1
        self.section = ['第六區']
        # ComboBox2
        self.sensors = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6',]
        # checkBox1
        self.ui.checkBox1.setText('溫度 (Temp)')
        # checkBox2
        self.ui.checkBox2.setText('濕度 (Humid)')
        # checkBox3
        self.ui.checkBox3.setText('光照量 (micromol)')
        self.para_names = ['temperature', 'humidity', 'light']
        self.colors = ["#D3D4", "#4B88A2", "#BB0A21"]
        self.time = time.time()

        labelStyle = {'color': '#000000', 'font-size': '14pt'}
        self.ui.graphicsView_2.setLabel('left', "Temperature & RH", units='Celsius & %', **labelStyle)
        self.ui.graphicsView_2.setLabel('right', "PAR", units='micro mol', **labelStyle)
        # self.ui.graphicsView_2.setLabel('bottom', "timestamps", units='s', **labelStyle)
        self.ui.graphicsView_2.setBackground('w')
        # ------------------------------------------------------------------------
        # self.date_axis = TimeAxisItem(orientation='bottom')
        # date_axis = pg.graphicsItems.DateAxisItem.DateAxisItem(orientation = 'bottom')
        # self.ui.graphicsView_2 = pg.PlotWidget(axisItems = {'bottom': self.date_axis})
        # ------------------------------------------------------------------------
        # self.ui.graphicsView_2 = pg.PlotWidget(axisItems={'bottom': self.date_axis})
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
        for n in range(len(getjsons['data'][-288*10:])):
            self.temp.append(getjsons['data'][n-288*10]['temperature'])
            self.humid.append(getjsons['data'][n-288*10]['wetness'])
            self.light.append(getjsons['data'][n-288*10]['par'])
            self.timestamp.append(getjsons['data'][n-288*10]['time'])
        # print(type(self.timestamp[0]))
        print(len(self.timestamp))
        print(len(self.temp))
        print(len(self.humid))
        print(len(self.light))
        print(self.time)
        self.paras = [self.temp, self.humid, self.light]
        self.show_Features()
    
    def show_Features(self):
        self.ui.graphicsView_2.clear()
        for n in range(len(self.checkBoxes)):
            if self.checkBoxes[n].isChecked():
                # self.date_axis = TimeAxisItem(orientation='bottom')
                # self.ui.graphicsView_2 = pg.PlotWidget(axisItems = {'bottom': self.date_axis})
                # self.checkBoxes[n].setBackground(self.colors[n])
                print("add {} feature".format(self.para_names[n]))
                self.ui.graphicsView_2.plot(self.timestamp, self.paras[n], pen=pg.mkPen(color=self.colors[n], width=3))
                # self.ui.graphicsView_2.setXRange(self.time-24*60*60*3, self.time)
                pg.QtGui.QApplication.processEvents()

if __name__ == '__main__':
     app = QtWidgets.QApplication([])
     window = MainWindow()
     window.show()
     sys.exit(app.exec_())