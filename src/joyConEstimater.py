import numpy as np
import threading
import math

from src.manager import JoyConManager
from src.estimator import PoseEstimator
from src.visualizer import Visualizer

class JoyConEstimator(object):
    '''JoyConEstimator

    JoyConの姿勢推定を実行するクラス．
    JoyConのデータを受け取り，Modelを指定したEstimatorで姿勢推定を実行する．
    '''

    def __init__(self,
                 model_name: str = "EKF",
                 apply_acc_correction: bool = True,
                 max_params_length: int = 1000
                 ):
        self.__estimator = PoseEstimator(model_name=model_name)
        self.__manager = JoyConManager()
        self.__base_time: float = 0.
        self.__times: list[float] = []
        self.__accs: list[np.ndarray] = []
        self.__gyros: list[np.ndarray] = []
        self.__xs: list[np.ndarray] = []
        self.__acc_coeff: float = 1.
        self.__apply_acc_correction: bool = apply_acc_correction
        self.__max_params_length: float = max_params_length
        self._stop_event = threading.Event()

    def connect(self):
        self.__manager.connect()
        print("JoyCon接続完了")

    def reset(self):
        self.__base_time: float = 0.
        self.__times: list[float] = []
        self.__accs: list[np.ndarray] = []
        self.__accs_global: list[np.ndarray] = []
        self.__acc_norms: list[float] = []
        self.__gyros: list[np.ndarray] = []
        self.__xs: list[np.ndarray] = []
        self.__estimator.reset()

    def start(self, show_chart: bool = True):
        print("測定開始")
        self.reset()
        viz:Visualizer = Visualizer()
        if show_chart:
            viz.prepare()
        while not self._stop_event.is_set():
            self.__manager.get_status()
            self.__accs += [self.__manager.acc*self.__acc_coeff]
            self.__acc_norms += [float(np.linalg.norm(self.__manager.acc))]
            self.__gyros += [self.__manager.gyro]
            if len(self.__accs) == 1:
                self.__base_time = self.__manager.time
                self.__times += [self.__manager.time - self.__base_time]
                self.__xs += [np.array([0., 0., 0.])]
            else:
                self.__times += [self.__manager.time - self.__base_time]
                dt = self.__times[-1] - self.__times[-2]
                self.__estimator.update(self.__manager.acc, self.__manager.gyro, dt)
                self.__xs += [self.__estimator.x*180./math.pi]
            R = self.__converter_to_global(self.__xs[-1])
            self.__accs_global += [R.dot(self.__accs[-1])]
            self.__acc_coeff = self.__get_acc_coeff()
            self.__trim_if_required()
            if show_chart:
                viz.update_chart(self.__xs, self.__accs, self.__accs_global, self.__times[-1])
        print("測定を終了しました．")

    def stop(self):
        print("測定を終了します．")
        self._stop_event.set()

    def __get_acc_coeff(self) -> float:
        if len(self.__times) % 50 != 0 or self.__apply_acc_correction == False:
            return 1.
        # print('norm: {}'.format(self.__acc_norms[-1]))
        # print('acc: {}'.format(self.__accs[-1]))
        # print('gyro: {}'.format(self.__gyros[-1]))
        # 50点ごとに変動係数を元に加速度補正
        # 加速度のばらつきが小さい=静止中の値を元に補正
        # 過去50点を取得
        arr = np.array(self.__acc_norms[-50:])
        # 平均
        mean = np.mean(arr)
        # 変動係数
        var = np.std(arr) / mean
        # print('変動係数: {}'.format(var))
        if var < 0.02 and (mean < 9.75 or 9.85 < mean):
            return 9.8 / float(np.linalg.norm(self.__accs[-1]))
        else:
            return 1.

    def __trim_if_required(self):
        if len(self.__times) > self.__max_params_length:
            self.__times = self.__times[-self.__max_params_length:]
            self.__accs = self.__accs[-self.__max_params_length:]
            self.__accs_global = self.__accs_global[-self.__max_params_length:]
            self.__acc_norms = self.__acc_norms[-self.__max_params_length:]
            self.__gyros = self.__gyros[-self.__max_params_length:]
            self.__xs = self.__xs[-self.__max_params_length:]

    @staticmethod
    def __converter_to_global(x):
        x = np.radians(x)
        c1 = np.cos(x[0])
        s1 = np.sin(x[0])
        c2 = np.cos(x[1])
        s2 = np.sin(x[1])
        c3 = np.cos(x[2])
        s3 = np.sin(x[2])
        Rx = np.array([
            [1, 0, 0],
            [0, c1, -s1],
            [0, s1, c1],
        ])
        Ry = np.array([
            [c2, 0, s2],
            [0, 1, 0],
            [-s2, 0, c2],
        ])
        Rz = np.array([
            [c3, -s3, 0],
            [s3, c3, 0],
            [0, 0, 1],
        ])
        Rxyz = Rz.dot(Ry).dot(Rx)
        return Rxyz