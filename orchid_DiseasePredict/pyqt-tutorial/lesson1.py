import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtCore import Qt

class MainWindow(QtWidgets, QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title will be passed to the function.
        self.windowTitleChanged.connect(self.onWindowTitleChange)

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title is discarded in the lambda and the
        # function is called without parameters.
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn())

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title is passed to the function
        # and replaces the default parameter
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn(x))

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title is passed to the function
        # and replaces the default parameter. Extra data is passed from
        # within the lambda.
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn(x, 25))
        
        self.setWindowTitle("My Awesome App")

        label = QLabel("This is a PyQt5 window!")

        label.setAlignment(Qt.AlignCenter)

        self.setCentralWidget(label)

def onWindowTitleChange(self, s):
    print(s)

def my_custom_fn(self, a="HELLO!", b=5):
    print(a, b)
app = QApplication(sys.argv)

window = MainWindow()
window.show()
# Start the event loop.
app.exec_()

# Your application won't reach here until you exit and the event
# loop has stopped.