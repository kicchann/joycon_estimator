from abc import ABC, abstractmethod
import numpy as np

class IModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @property
    @abstractmethod
    def x(self):
        pass

    @property
    @abstractmethod
    def P(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self,
               gyro: np.ndarray,
               acc: np.ndarray,
               delta_t: float
               ):
        pass

