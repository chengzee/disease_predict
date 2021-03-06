# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\V1.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from datetime import datetime

class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(int(values[n]/1000)) for n in range(len(values))]

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1352, 523)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 60, 551, 411))
        self.groupBox.setObjectName("groupBox")
        self.graphicsView = QtWidgets.QGraphicsView(self.groupBox)
        self.graphicsView.setGeometry(QtCore.QRect(10, 20, 531, 381))
        self.graphicsView.setObjectName("graphicsView")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(570, 40, 21, 421))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.comboBox2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox2.setGeometry(QtCore.QRect(730, 30, 69, 22))
        self.comboBox2.setObjectName("comboBox2")
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(640, 30, 81, 21))
        self.label2.setObjectName("label2")
        self.checkBox1 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox1.setGeometry(QtCore.QRect(870, 30, 121, 16))
        self.checkBox1.setObjectName("checkBox1")
        # ----------------------------------------------------
        self.date_axis = TimeAxisItem(orientation='bottom')
        # ----------------------------------------------------
        self.graphicsView_2 = pg.PlotWidget(self.centralwidget, axisItems = {'bottom': self.date_axis})
        self.graphicsView_2.setGeometry(QtCore.QRect(610, 80, 721, 381))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.checkBox2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox2.setGeometry(QtCore.QRect(990, 30, 121, 16))
        self.checkBox2.setObjectName("checkBox2")
        self.checkBox3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox3.setGeometry(QtCore.QRect(1110, 30, 121, 16))
        self.checkBox3.setObjectName("checkBox3")
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(30, 20, 171, 31))
        self.label1.setObjectName("label1")
        self.comboBox1 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox1.setGeometry(QtCore.QRect(250, 30, 69, 22))
        self.comboBox1.setObjectName("comboBox1")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1352, 21))
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
        self.checkBox1.setText(_translate("MainWindow", "CheckBox"))
        self.checkBox2.setText(_translate("MainWindow", "CheckBox"))
        self.checkBox3.setText(_translate("MainWindow", "CheckBox"))
        self.label1.setText(_translate("MainWindow", "TextLabel"))
