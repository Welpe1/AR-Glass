import sys
from PyQt6 import QtCore,QtWidgets
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton,QHBoxLayout,QVBoxLayout
from PyQt6.QtCore import Qt,QTimer,pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor
from hand_gesture import hand_connections
import time

class FuncBaseWidget(QWidget):
    def __init__(self,
                  parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        button_style="""
            QPushButton {
                border-radius: 20px;
                background-color: rgba(0, 0, 0, 150);
                color: white;
                font-size: 60px;
                border: 2px solid rgba(255, 255, 255, 100);
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            #back_btn {
                min-width: 200px;
                max-width: 200px;
                min-height: 100px;
                max-height: 100px;
                border-radius: 10px;
            }
        """
        # 主布局
        main_layout=QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.addStretch()
       
        self.back_btn=QtWidgets.QPushButton("返回")
        self.back_btn.setFixedSize(200, 100)
        self.back_btn.setStyleSheet(button_style)
        self.back_btn.setObjectName("back_btn")
        main_layout.addWidget(self.back_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        main_layout.addStretch()
    

class MainContent(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        # 按钮样式
        button_style = """
        QPushButton {
            border-radius: 20px;
            background-color: rgba(0, 0, 0, 150);
            color: white;
            font-size: 53px;
            min-width: 200px;
            min-height: 200px;
            max-width: 200px;
            max-height: 200px;
            border: 2px solid rgba(255, 255, 255, 100);
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:pressed {
            background-color: #3e8e41;
        }
        """

        # 主网格布局（固定间距50px）
        grid_layout = QGridLayout(self)
        grid_layout.setSpacing(50)  # 按钮间距固定50px
        grid_layout.setContentsMargins(0, 0, 0, 0)

        # 创建6个按钮
        self.Image_recognition = QPushButton("图像识别")
        self.Text_recognition = QPushButton("文字识别")
        self.Speech_recognition = QPushButton("语音识别")
        self.Remote_control = QPushButton("远程控制")
        self.Remote_camera = QPushButton("远程串流")
        self.Photograph = QPushButton("拍照")

        # 设置按钮样式
        for btn in [
            self.Image_recognition,
            self.Text_recognition,
            self.Speech_recognition,
            self.Remote_control,
            self.Remote_camera,
            self.Photograph
        ]:
            btn.setStyleSheet(button_style)

        # 添加到网格布局（3x2排列）
        grid_layout.addWidget(self.Image_recognition, 0, 0)
        grid_layout.addWidget(self.Text_recognition, 0, 1)
        grid_layout.addWidget(self.Speech_recognition, 0, 2)
        grid_layout.addWidget(self.Remote_control, 1, 0)
        grid_layout.addWidget(self.Remote_camera, 1, 1)
        grid_layout.addWidget(self.Photograph, 1, 2)

class Widget_image(FuncBaseWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        

class Widget_text(FuncBaseWidget):
    def __init__(self,parent=None):
        super().__init__(parent)

class Widget_speech(FuncBaseWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

class Widget_control(FuncBaseWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        

class Widget_camera(FuncBaseWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        



class Widget_main(QWidget):
    func_signal=pyqtSignal(int)#定义信号
    def __init__(self):
        super().__init__()
        self.window_width = 1500
        self.window_height = 900
        
        # 初始化UI
        self.init_ui()
        
        # 手势参数
        self.is_click = False
        self.landmarks = None
        self.last_click_time = 0
        
        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_button_click)
        self.timer.start(30)

    def init_ui(self):
        """初始化主界面"""
        self.setGeometry(100, 100, self.window_width, self.window_height)
        
        # 主布局
        self.main_layout = QtWidgets.QStackedLayout()
        self.setLayout(self.main_layout)
        

        self.main_content = MainContent()
        self.image_content = Widget_image()
        self.text_content = Widget_text()
        self.speech_content = Widget_speech()
        self.control_content = Widget_control()
        self.camera_content = Widget_camera()
        # self.photograph_content = Widget_photograph()
        
        # 添加到堆叠布局
        self.main_layout.addWidget(self.main_content)
        self.main_layout.addWidget(self.image_content)
        self.main_layout.addWidget(self.text_content)
        self.main_layout.addWidget(self.speech_content)
        self.main_layout.addWidget(self.control_content)
        self.main_layout.addWidget(self.camera_content)
        # self.main_layout.addWidget(self.photograph_content)
        
        # 连接信号
        self.main_content.Image_recognition.clicked.connect(self.show_image_content)
        self.main_content.Text_recognition.clicked.connect(self.show_text_content)
        self.main_content.Speech_recognition.clicked.connect(self.show_speech_content)
        self.main_content.Remote_control.clicked.connect(self.show_control_content)
        self.main_content.Remote_camera.clicked.connect(self.show_camera_content)
        # self.main_content.Photograph.clicked.connect(self.show_photograph_content)
        self.image_content.back_btn.clicked.connect(self.show_main_content)
        self.text_content.back_btn.clicked.connect(self.show_main_content)
        self.speech_content.back_btn.clicked.connect(self.show_main_content)
        self.control_content.back_btn.clicked.connect(self.show_main_content)
        self.camera_content.back_btn.clicked.connect(self.show_main_content)
        
        # 设置背景
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.GlobalColor.black)
        self.setPalette(palette)
        
        # 初始化画笔
        self.pen = QPen(QColor(255, 50, 50, 100), 2, Qt.PenStyle.DashLine)

    def show_main_content(self):
        self.main_layout.setCurrentIndex(0)
        from PyQt6.QtCore import Qt,QTimer,pyqtSignal
        print("main")

    def show_image_content(self):
        self.main_layout.setCurrentIndex(1)
        self.func_signal.emit(2)
    
    def show_text_content(self):
        self.main_layout.setCurrentIndex(2)

    def show_speech_content(self):
        self.main_layout.setCurrentIndex(3)
        self.func_signal.emit(1)  #发送信号

    def show_control_content(self):
        self.main_layout.setCurrentIndex(4)

    def show_camera_content(self):
        self.main_layout.setCurrentIndex(5)

    def show_photograph_content(self):
        self.main_layout.setCurrentIndex(6)


    def update_gesture(self, is_click, landmarks):
        """更新手势数据"""
        self.is_click=is_click
        self.landmarks=landmarks
        self.update()

    def check_button_click(self):
        """检测手势点击"""
        if not self.is_click or not self.landmarks or len(self.landmarks) < 9:
            return
            
        # 防抖处理
        current_time = time.time()
        if current_time - self.last_click_time < 0.5:
            return
            
        # 获取食指尖端位置
        index_tip = self.landmarks[8]
        x_pos = index_tip[0] * self.width()
        y_pos = index_tip[1] * self.height()
        
        # 根据当前页面检测不同按钮
        if self.main_layout.currentIndex() == 0:  # 主界面
            buttons = [
                self.main_content.Image_recognition,
                self.main_content.Text_recognition,
                self.main_content.Speech_recognition,
                self.main_content.Remote_control,
                self.main_content.Remote_camera,
                self.main_content.Photograph
            ]
        elif self.main_layout.currentIndex() == 1:  # 图像识别界面
            buttons = [
                self.image_content.back_btn
            ]
        elif self.main_layout.currentIndex() == 2: 
            buttons = [
                self.text_content.back_btn
            ]
        elif self.main_layout.currentIndex() == 3: 
            buttons = [
                self.text_content.back_btn
            ]
        elif self.main_layout.currentIndex() == 4: 
            buttons = [
                self.text_content.back_btn
            ]
        elif self.main_layout.currentIndex() == 5: 
            buttons = [
                self.text_content.back_btn
            ]
        elif self.main_layout.currentIndex() == 6: 
            buttons = [
                self.text_content.back_btn
            ]
        
        for btn in buttons:
            if self.is_point_in_button(x_pos, y_pos, btn):
                btn.click()
                self.last_click_time = current_time
                break

    def is_point_in_button(self, x, y, button):
        """检查坐标是否在按钮区域内"""
        btn_pos = button.mapTo(self, QtCore.QPoint(0, 0))
        btn_rect = QtCore.QRect(btn_pos, button.size())
        return btn_rect.contains(int(x), int(y))

    def paintEvent(self,event):
        painter=QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        
        if self.landmarks:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            width, height = self.width(), self.height()
            painter.setPen(self.pen)
            # 绘制关键点
            for landmark in self.landmarks:
                x=int(landmark[0]*width)
                y=int(landmark[1]*height)
                painter.drawEllipse(x,y,5,5)
            
            # # 绘制连接线
            # for connection in hand_connections:
            #     if(connection[0] < len(self.landmarks) and (connection[1] < len(self.landmarks))):
            #         x1 = int(self.landmarks[connection[0]][0] * width)
            #         y1 = int(self.landmarks[connection[0]][1] * height)
            #         x2 = int(self.landmarks[connection[1]][0] * width)
            #         y2 = int(self.landmarks[connection[1]][1] * height)
            #         painter.drawLine(x1, y1, x2, y2)
        
        painter.end()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Widget_main()
    window.show()
    sys.exit(app.exec())