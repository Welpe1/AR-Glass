python3.12环境
windows安装了pyqt5
pip install pyqt5 

linux开发板安装pyqt5/6
有时候是5有时候是6
conda install -c conda-forge pyqt

阿里云模型的API-Key

建议绑定到环境变量中
echo "export DASHSCOPE_API_KEY='sk-a5b547ef397246efa14ab7a737e6fb1c'" >> ~/.bashrc
source ~/.bashrc
echo $DASHSCOPE_API_KEY



W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning message.)
W rknn-toolkit-lite2 version: 2.1.0
E Invalid RKNN model path: ./model/ppocrv4_det_rk3588_int8.rknn
E Model is not loaded yet, this interface should be called after load_rknn!
W rknn-toolkit-lite2 version: 2.1.0
E Invalid RKNN model path: ./model/ppocrv4_rec_rk3588_fp16.rknn
E Model is not loaded yet, this interface should be called after load_rknn!
Traceback (most recent call last):
  File "/home/elf/workspace/ar-host/main.py", line 22, in <module>
    func_thread=FuncThread(camera)
                ^^^^^^^^^^^^^^^^^^
  File "/home/elf/workspace/ar-host/func_choose.py", line 29, in __init__
    raise RuntimeError("无法附加到共享内存请确保写入线程已启动")
RuntimeError: 无法附加到共享内存请确保写入线程已启动
