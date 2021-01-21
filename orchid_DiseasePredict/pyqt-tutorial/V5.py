# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'V5.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from datetime import datetime

class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(int(values[n]/1000)) for n in range(len(values))]

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1330, 791)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 60, 531, 321))
        self.groupBox.setObjectName("groupBox")
        self.graphicsView = QtWidgets.QGraphicsView(self.groupBox)
        self.graphicsView.setGeometry(QtCore.QRect(10, 20, 511, 281))
        self.graphicsView.setObjectName("graphicsView")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(570, 40, 21, 691))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.comboBox2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox2.setGeometry(QtCore.QRect(680, 10, 69, 22))
        self.comboBox2.setObjectName("comboBox2")
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(620, 10, 81, 21))
        self.label2.setObjectName("label2")
        # ----------------------------------------------------
        self.date_axis_1 = TimeAxisItem(orientation='bottom')
        self.date_axis_2 = TimeAxisItem(orientation='bottom')
        self.date_axis_3 = TimeAxisItem(orientation='bottom')
        # ----------------------------------------------------
        self.graphicsView_2 = pg.PlotWidget(self.centralwidget, axisItems = {'bottom': self.date_axis_1})
        self.graphicsView_2.setGeometry(QtCore.QRect(610, 50, 661, 221))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(30, 20, 171, 31))
        self.label1.setObjectName("label1")
        self.comboBox1 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox1.setGeometry(QtCore.QRect(250, 30, 69, 22))
        self.comboBox1.setObjectName("comboBox1")
        self.voltage_meter = QtWidgets.QLabel(self.centralwidget)
        self.voltage_meter.setGeometry(QtCore.QRect(850, 10, 131, 20))
        self.voltage_meter.setObjectName("voltage_meter")
        self.update_time = QtWidgets.QLabel(self.centralwidget)
        self.update_time.setGeometry(QtCore.QRect(1090, 10, 221, 31))
        self.update_time.setObjectName("update_time")
        self.graphicsView_3 = pg.PlotWidget(self.centralwidget, axisItems = {'bottom': self.date_axis_2})
        self.graphicsView_3.setGeometry(QtCore.QRect(610, 280, 661, 221))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 380, 531, 321))
        self.groupBox_2.setObjectName("groupBox_2")
        self.graphicsView_7 = QtWidgets.QGraphicsView(self.groupBox_2)
        self.graphicsView_7.setGeometry(QtCore.QRect(10, 20, 511, 281))
        self.graphicsView_7.setObjectName("graphicsView_7")
        self.graphicsView_4 = pg.PlotWidget(self.centralwidget, axisItems = {'bottom': self.date_axis_3})
        self.graphicsView_4.setGeometry(QtCore.QRect(610, 510, 661, 221))
        self.graphicsView_4.setObjectName("graphicsView_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1330, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.label2.setText(_translate("MainWindow", "TextLabel"))
        self.label1.setText(_translate("MainWindow", "TextLabel"))
        self.voltage_meter.setText(_translate("MainWindow", "TextLabel"))
        self.update_time.setText(_translate("MainWindow", "TextLabel"))
        self.groupBox_2.setTitle(_translate("MainWindow", "GroupBox"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

