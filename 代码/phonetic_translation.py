import os
from typing import Optional
import pyaudio
import dashscope
from dashscope.audio.asr import *
from PyQt6.QtCore import pyqtSignal,QThread

def api_key_init():
    if 'DASHSCOPE_API_KEY' in os.environ:
        dashscope.api_key = os.environ[
        'DASHSCOPE_API_KEY'] 
    else:
        dashscope.api_key = 'sk-a5b547ef397246efa14ab7a737e6fb1c'  

class TranslationCallback(TranslationRecognizerCallback):
    def __init__(self,
                 target_Language:str='zh'):
        
        #类注解,表明self.mic可以是pyaudio.PyAudio或None
        self.mic:Optional[pyaudio.PyAudio]=None
        self.stream:Optional[pyaudio.Stream]=None
        self.target_language=target_Language

    #服务器连接成功时
    def on_open(self)->None:
        try:
            self.mic=pyaudio.PyAudio()
            self.stream=self.mic.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=16000,
                                    input=True)
        except Exception as e:
            raise RuntimeError(f"audio init fail:{str(e)}")
        
    #翻译服务关闭时
    def on_close(self)->None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.mic:
            self.mic.terminate()

    #发送错误时
    def on_error(self,error:Exception)->None:
        self.on_close()
    
    #收到翻译结果时
    def on_event(self, 
                 request_id,
                 transcription_result,
                 translation_result:TranslationResult,
                 usage)->None:
        #翻译结果
        if translation_result:
            try:
                translation=translation_result.get_translation(self.target_language)
                if translation.is_sentence_end:
                    print('translate to {}: {}'.format(self.target_language,translation.text))
            except Exception as e:
                print(f"translate result error:{str(e)}")


class TranslationThread(QThread):
    def __init__(self,
                 target_language:str='zh'):
        super().__init__()

        self.target_language=target_language
        self.callback=None
        self.translator=None
        self.isRunning=False

    def run(self):
        api_key_init()
        self.callback=TranslationCallback(self.target_language)
        self.translator=TranslationRecognizerRealtime(
            model='gummy-realtime-v1',  #指定模型
            format='pcm',               #音频格式
            sample_rate=16000,          #采样率
            transcription_enabled=False,#禁用转录（仅翻译）
            translation_enabled=True,   #启用翻译
            translation_target_languages=[self.target_language],  #目标语言（中文）
            callback=self.callback,          # 设置回调处理器
        )
        self.translator.start()
        self.isRunning=True
        while self.isRunning and self.callback.stream:
                data=self.callback.stream.read(3200,exception_on_overflow=False)
                self.translator.send_audio_frame(data)
        self.stop()

    def stop(self):
        self.isRunning=False
        if self.translator:
            self.translator.stop()
        if self.callback:
            self.callback.on_close()






