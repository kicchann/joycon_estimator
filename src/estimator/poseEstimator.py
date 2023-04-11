import numpy as np
from src.estimator.models import ExtendedKalmanFilterModel

class PoseEstimator(object):
    '''PoseExtimator

    姿勢推定を実行するクラス．
    推定モデルを受け取り，実行する．

    TODO:
        将来的に複数モデルを受け取れるようにする．

    Attributes:
        x (np.ndarray): オイラー角
        P (np.ndarray): 事前共分散行列

    Methods:
        reset: 姿勢の初期化
        update: 姿勢の更新


    '''

    def __init__(self, model_name: str = "EKF"):
        models = {
            "EKF": ExtendedKalmanFilterModel
        }
        if model_name not in models.keys():
            raise ValueError("model_name is not supported.")
        self.__estimator = models[model_name]()

    @property
    def x(self) -> np.ndarray:
        return self.__estimator.x

    @property
    def P(self) -> np.ndarray:
        return self.__estimator.P

    def reset(self):
        '''reset

        姿勢の初期化
        '''
        self.__estimator.reset()

    def update(self,
               acc: np.ndarray,
               gyro: np.ndarray,
               delta_time: float,
               ):
        '''update

        姿勢の更新

        Args:
            acc (np.ndarray): 加速度の観測値
            gyro (np.ndarray): ジャイロの観測値
            delta_time (float): 前回の更新からの時間ステップ
        '''
        self.__estimator.update(acc, gyro, delta_time)