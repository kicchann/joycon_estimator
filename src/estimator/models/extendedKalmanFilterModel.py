from .iModel import IModel
import numpy as np
from typing import Optional
import math

class ExtendedKalmanFilterModel(IModel):
    '''ExtendedKalmanModel

    拡張カルマンフィルタの実行クラス
    以下を参考に作成
        https://qiita.com/yakiimo121/items/97f3c174e0d0db74535a
        https://memo.soarcloud.com/6%E8%BB%B8imu%EF%BD%9E%E6%8B%A1%E5%BC%B5%E3%82%AB%E3%83%AB%E3%83%9E%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF/

    attributes:
        x (np.ndarray): オイラー角 [phi, theta, psi]の推定値
        P (np.ndarray): 事前共分散行列の推定値

    methods:
        reset(x: Optional[np.ndarray] = None, P: Optional[np.ndarray] = None)
            オイラー角xと，事前共分散行列Pを初期化する．
        update(gyro: np.ndarray, acc: np.ndarray, delta_t: float)
            拡張カルマンフィルタによるオイラー角xと，事前共分散行列Pの更新を行う．
    '''

    def __init__(self):
        '''__init__

        コンストラクタ．

        Args:
            x (np.ndarray): オイラー角 [phi, theta, psi]の初期値
            P (np.ndarray): 事前共分散行列の初期値
            gyro (np.ndarray): ジャイロセンサの値
            acc (np.ndarray): 加速度センサの値
            delta_t (float): サンプリング周期
        
        methods:
            reset(x: Optional[np.ndarray] = None, P: Optional[np.ndarray] = None)
                オイラー角xと，事前共分散行列Pを初期化する．
            update(gyro: np.ndarray, acc: np.ndarray, delta_t: float)
                拡張カルマンフィルタによるオイラー角xと，事前共分散行列Pの更新を行う．
        '''
        self.__var_coeff: float = 0.0174
        self.__delta_t: float = 0.01
        self.__x: np.ndarray
        self.__P: np.ndarray
        self.__gyro: np.ndarray
        self.__acc: np.ndarray


    def __str__(self):
        return 'ExtendedKalmanFilterModel'

    @property
    def x(self):
        return self.__x

    @property
    def P(self):
        return self.__P

    def reset(self):
        self.__x = np.array([0., 0., 0.])
        self.__P = np.eye(3) * self.__var_coeff * (self.__delta_t)*2

    def update(self,
               acc: np.ndarray,
               gyro: np.ndarray,
               delta_t: float
               ):
        '''update

        拡張カルマンフィルタによるオイラー角xと，事前共分散行列Pの更新を行う．

        Args:
            gyro (np.ndarray): ジャイロの観測値 [wx,wy,wz]
            acc (np.ndarray): 加速度の観測値 [ax,ay,az]
            delta_t (float): 時間ステップ

        '''
        self.__acc = acc
        self.__gyro = gyro
        self.__delta_t = delta_t
        ### 予測値の計算 ###
        # オイラー角の予測値(オイラー角とジャイロから計算)
        x_pred = self.__x + self.__delta_euler
        # 事前共分散行列の予測値(オイラー角とジャイロから計算)
        F = self.__jacobian_F
        P_pred: np.ndarray = F @ self.__P @ F.T + self.__Q

        ### 更新値の計算 ###
        # 状態観測値(加速度から計算)
        y = self.__euler_by_acc
        # 状態観測値から，応答値を計算
        y_res = y - np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) @ x_pred
        # カルマンゲインKの計算
        S = self.__H @ P_pred @ self.__H.T + self.__R
        K = P_pred @ self.__H.T @ np.linalg.inv(S)
        # オイラー角の更新値
        self.__x = x_pred + K @ y_res[:2]
        # 事前共分散行列の更新値
        self.__P = (np.eye(3) - K @ self.__H) @ P_pred

    @property
    def __H(self) -> np.ndarray:
        return np.eye(2, 3)

    @property
    def __Q(self) -> np.ndarray:
        '''__Q

        オイラー角の観測誤差期待値
        '''
        return np.eye(3) * self.__var_coeff * (self.__delta_t**2)

    @property
    def __R(self) -> np.ndarray:
        '''__R

        加速度の観測誤差期待値
        '''
        return np.eye(2) * (self.__delta_t**2)

    @property
    def __delta_euler(self) -> np.ndarray:
        '''__get_delta_euler

        微小時間におけるオイラー角の変化量を計算する．
        de = dt * de/dt

        Returns:
            np.ndarray: 微小時間におけるオイラー角の変化量
        '''
        phi: float = self.__x[0]
        theta: float = self.__x[1]
        psi: float = self.__x[2]
        m = np.array(
            [[1, math.sin(phi)*math.tan(theta), math.cos(phi)*math.tan(theta)],
             [0, math.cos(phi), -math.sin(phi)],
             [0, math.sin(phi)/math.cos(theta), math.cos(phi)/math.cos(theta)]] # type: ignore
        )
        return self.__delta_t * m.dot(self.__gyro.T).T

    @property
    def __jacobian_F(self) -> np.ndarray:
        '''__get_jacobian_F

        ヤコビアン行列を計算する．

        Args:
            x (np.ndarray): t-1でのオイラー角
            gyro (np.ndarray): ジャイロの観測値
            delta_t (float): 時間ステップ

        Returns:
            np.ndarray: ヤコビアン行列
        '''

        wx: float = self.__gyro[0]
        wy: float = self.__gyro[1]
        wz: float = self.__gyro[2]
        phi: float = self.__x[0]
        theta: float = self.__x[1]
        psi: float = self.__x[2]
        c = math.cos
        s = math.sin
        t = math.tan
        a11 = 1. + self.__delta_t*(wy*c(phi)*t(theta)-wz*s(phi)*t(theta))
        a12 = self.__delta_t*(wy*s(phi)/(c(theta)**2)+wz*c(phi)/(c(theta)**2))
        a13 = 0.
        a21 = -self.__delta_t*(wy*s(phi)-wz*c(phi))
        a22 = 1.
        a23 = 0.
        a31 = wy*c(phi)/(c(theta))-wz*s(phi)/(c(theta))
        a32 = wy*s(phi)*(t(theta)**2)+wz*c(phi)*(t(theta)**2)
        a33 = 1.
        return np.array(
            [[a11, a12, a13],
             [a21, a22, a23],
             [a31, a32, a33]]
        )

    @property
    def __euler_by_acc(self) -> np.ndarray:
        '''__get_euler_by_acc

        加速度から，オイラー角を計算する．

        Args:
            acc (np.ndarray): 加速度の観測値

        Returns:
            np.ndarray: オイラー角
        '''
        ax: float = self.__acc[0]
        ay: float = self.__acc[1]
        az: float = self.__acc[2]
        return np.array([
            math.atan(ay / (az)),
            math.atan(ax / (math.sqrt(ay**2 + az**2))),
            0.
        ])