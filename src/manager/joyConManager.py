import numpy as np
from typing import Optional
import math
import time

from pyjoycon import device
from pyjoycon.joycon import JoyCon

class JoyConManager(object):
    '''JoyconManager

    JoyConのデータを管理するクラス．
    JoyConと接続して，加速度とジャイロの値を取得する．

    JoyConとの接続には以下のライブラリを使用
        https://github.com/tocoteron/joycon-python
    補正の情報などは以下を参考
        https://github.com/dekuNukem/Nintendo_Switch_Reverse_Engineering
    '''

    def __init__(self):
        self.__ids: list = []
        self.__joycon: Optional[JoyCon] = None

        self.__time_now: float = 0.
        self.__acc: np.ndarray = np.zeros(3)
        self.__gyro: np.ndarray = np.zeros(3)

    @property
    def acc(self) -> np.ndarray:
        return self.__acc

    @property
    def gyro(self) -> np.ndarray:
        return self.__gyro

    @property
    def time(self) -> float:
        return self.__time_now

    def connect(self):
        '''connect_joycon

        JoyConと接続する．
        '''
        try:
            self.__ids: list = device.get_device_ids()[0]
            self.__joycon = JoyCon(*self.__ids)
        except IndexError:
            print('JoyConが接続されていません．')
            exit(1)

    def get_status(self):
        status: dict = self.__joycon.get_status()
        assert status is not None
        # 時間の取得
        self.__time_now = time.perf_counter()
        self.__acc = np.array([float(v) for v in status["accel"].values()])
        # x軸offset補正
        self.__acc[0] -= 350.
        # 重力加速度に変換
        self.__acc *= 0.000224*9.8

        self.__gyro = np.array([float(v) for v in status["gyro"].values()])
        # ラジアンに変換
        self.__gyro *= 0.06103/180.*math.pi