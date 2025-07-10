import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hand_gesture import Camera,GestureThread
from func_choose import FuncThread
from widget_main import Widget_main
from PyQt6.QtWidgets import QApplication


#QT显示界面为主线程
#手势识别子线程
#语音识别子线程
#图像识别子线程
#网络通信子线程


if __name__ == "__main__":
    app=QApplication(sys.argv)
    camera=Camera(0)

    gesture_thread=GestureThread(camera)
    # func_thread=FuncThread(camera)
    window=Widget_main()

    gesture_thread.gesture_signal.connect(window.update_gesture)
    # gesture_thread.shm_send_finish.connect(func_thread.read_from_shm)
    # window.func_signal.connect(func_thread.choose_func)
    # func_thread.shm_need_send.connect(gesture_thread.shm_need_send)

    gesture_thread.start()  # 先启动线程
    # func_thread.start()
    window.show()

    sys.exit(app.exec())

    gesture_thread.stop()
    func_thread.stop()
