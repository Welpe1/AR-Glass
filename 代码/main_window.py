import numpy as np
from hand_gesture import hand_connections
from PyQt6.QtWidgets import  QMainWindow,QWidget,QPushButton,QVBoxLayout
from PyQt6.QtCore import Qt,QTimer,pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor


#qt主窗口,主线程运行
class MainWindow(QMainWindow):
    func_signal=pyqtSignal(int)#定义信号
    def __init__(self,
                 window_width=1280,
                 window_height=960):
        super().__init__()
        self.window_width=window_width
        self.window_height=window_height
        #鼠标参数

        self.is_click=False
        self.landmarks=None

        self.init_ui()
        self.timer=QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(25)

    def init_ui(self):
        self.setGeometry(100,100,self.window_width,self.window_height)
        central_widget=QWidget()
        self.setCentralWidget(central_widget)
        self.pen=QPen(QColor(255,50,50,100),2,Qt.PenStyle.DashLine)  #画笔(颜色RGBA,粗细,线条样式(实线,点线等))
        #设置背景为黑色
        self.setAutoFillBackground(True)
        p=self.palette()#获取部件调色板
        p.setColor(self.backgroundRole(),Qt.GlobalColor.black)
        self.setPalette(p)#将修改后的调色板重新设置到部件

    def update_gesture(self,is_click,landmarks):
        self.is_click,self.landmarks=is_click,landmarks
        if self.is_click:
            print("click")
            self.func_signal.emit(2)  #发送信号



    
    #30ms定时器触发界面更新
    def update_ui(self):
        try:
            self.update()
        except Exception as e:
            print(f"错误: {e}")
            self.close()
    
    def paintEvent(self,event):
        painter=QPainter(self)
        painter.fillRect(self.rect(),Qt.GlobalColor.black)
        if self.landmarks:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)  # 抗锯齿
            width,height=self.width(),self.height()
            painter.setPen(self.pen)
            for landmark in self.landmarks:
                x=int(landmark[0]*width)
                y=int(landmark[1]*height)
                painter.drawEllipse(x,y,5,5)
        
            # for connection in hand_connections:
            #     start_idx,end_idx=connection
            #     if start_idx<len(self.landmarks) and end_idx<len(self.landmarks):
            #         x1=int(self.landmarks[start_idx][0]*width)
            #         y1=int(self.landmarks[start_idx][1]*height)
            #         x2=int(self.landmarks[end_idx][0]*width)
            #         y2=int(self.landmarks[end_idx][1]*height)
            #         painter.drawLine(x1,y1,x2,y2)
        painter.end()

    def closeEvent(self,event):
        event.accept()







