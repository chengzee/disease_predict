import sys
import cv2
# from test_ui import Ui_MainWindow
from ui_0125 import Ui_MainWindow
# from test_ui_3 import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QListWidget, QLabel

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPalette

class Controller(QMainWindow, Ui_MainWindow):

    def __init__(self,parent =None):
        super(QMainWindow,self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_2.clicked.connect(self.buttonClicked_2)
        self.ui.pushButton_3.clicked.connect(self.buttonClicked)
        self.show_image()
        
    
    @pyqtSlot()
    def update(self):
        self.count+=1
        # self.click.seeText("您好")


    
    def confirmcheckbox(self):
        # check_list = [str(self.ui.checkBox.text()),self.ui.checkBox_2.text(),self.ui.checkBox_3.text(),self.ui.checkBox_4.text(),self.ui.checkBox_5()]
        # check_list = str(self.ui.checkBox.text())
        check_list = "aa"
        return (check_list)
        # confirm_list = [confirm_1,confirm_2,confirm_3,confirm_4,confirm_5]
        # disease_str = ""
        # for i in range(5):
            
        #     confirm_list[i] = self.ui.checkBox.isChecked()
        #     if(confirm_list[i] == 1):
        #         disease_str = disease_str + str(check_list[i])
        # return disease_str

    def show_image(self):
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(-600, -600, 1200, 1200)
        pic = QtGui.QPixmap("C:/Users/ICMEMS_I7-3770/darknet-master/build/darknet/x64/mar_s62.png")
        scene.addItem(QtGui.QGraphicsPixmapItem(pic))
        # self.ui.graphicsView = self.gv
        self.ui.graphicsView.setScene(scene)
        self.ui.graphicsView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.ui.graphicsView.show()

        
    
    # def buttonClicked_3(self):


    
    def buttonClicked_2(self):
        # predict_1 = 1
        # predict_4 = 1
        self.ui.checkBox_6.setChecked(True)
        self.ui.checkBox_7.setChecked(True)
        self.ui.label_5.setStyleSheet("background-color: rgb(255, 0, 0);")
        self.ui.checkBox.setStyleSheet("background-color: rgb(255, 0, 0);")
        self.ui.checkBox_2.setStyleSheet("background-color: rgb(255, 0, 0);")

        
    
    def buttonClicked(self):
        datetext = "2020/11/5 批號: "
        # text = datetext + self.ui.lineEdit.text()  +self.ui.checkBox.text()
        # text = datetext + self.ui.lineEdit.text()  +str(self.ui.checkBox.isChecked())
        # xx = str(self.confirmcheckbox)
        
        xxx_1 = self.ui.checkBox.text()
        xxx_2 = self.ui.checkBox_2.text()
        xxx_3 = self.ui.checkBox_3.text()
        xxx_4 = self.ui.checkBox_4.text()
        xxx_5 = self.ui.checkBox_5.text()

        check_text = ""
        
        if(self.ui.checkBox.isChecked() == True):
            if(self.ui.checkBox_6.isChecked()==True):
                check_text = " "+check_text+ xxx_1  
            
          
        if(self.ui.checkBox_2.isChecked() == True):
            if(self.ui.checkBox_7.isChecked()==True):
                check_text = check_text +" "+ xxx_2
        if(self.ui.checkBox_3.isChecked()==True):
            if(self.ui.checkBox_8.isChecked()==True):
                check_text = check_text+" "+xxx_3 
        if(self.ui.checkBox_4.isChecked() == True): 
            if(self.ui.checkBox_9.isChecked()==True):
                check_text = check_text+" "+xxx_4 
        if(self.ui.checkBox_5.isChecked()== True):
            if(self.ui.checkBox_10.isChecked()==True):
                check_text = check_text+" "+xxx_5
        
        self.ui.label_5.setStyleSheet("")
        self.ui.checkBox.setStyleSheet("")
        self.ui.checkBox_2.setStyleSheet("")
        self.ui.checkBox_3.setStyleSheet("")
        self.ui.checkBox_4.setStyleSheet("")
        self.ui.checkBox_5.setStyleSheet("")

        print(check_text)
        text = datetext + self.ui.lineEdit.text() + check_text
        # + confirmcheckbox(self.ui.checkBox_2)
        self.ui.listWidget.addItem(text)
        self.ui.checkBox_6.setChecked(0)
        self.ui.checkBox_7.setChecked(0)
        self.ui.checkBox_8.setChecked(0)
        self.ui.checkBox_9.setChecked(0)
        self.ui.checkBox_10.setChecked(0)
       
        


        

        # self.ui.label.setText(text)
        #
        # self.ui.lineEdit.clear()
    

  
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Controller()
    window.show()
    sys.exit(app.exec_())