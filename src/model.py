import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class LinearRegressionResult:
    w: float
    b: float
    cost_history: List[float]
    parameter_history: List[List[float]]

    def predict(self, x: float) -> float:
        return self.w * x + self.b

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        return self.w * x + self.b

class LinearRegressionModel:
    
    def __init__(self, w_init: float = 0, b_init: float = 0):
        self.w = w_init
        self.b = b_init
        self._cost_history = []
        self._parameter_history = []

    def _compute_y_hat(self, x: float, w: float, b: float) -> float:
        return w * x + b

    def _compute_error(self, y_hat: float, y: float) -> float:
        return (y_hat - y) ** 2

    def compute_cost(self, x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
        m = x.shape[0]
        cost = 0
        
        for i in range(m):
            f_wb = self._compute_y_hat(x[i], w, b)
            cost = cost + self._compute_error(f_wb, y[i])
        total_cost = 1 / (2 * m) * cost

        return total_cost

    def compute_gradient(self, x: np.ndarray, y: np.ndarray, w: float, b: float) -> Tuple[float, float]:
        m = x.shape[0]
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            f_wb = w * x[i] + b
            dj_dw_i = (f_wb - y[i]) * x[i]
            dj_db_i = f_wb - y[i]
            dj_dw += dj_dw_i
            dj_db += dj_db_i
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db

    def gradient_descent(self, x_train: np.ndarray, y_train: np.ndarray, 
            iterations: int = 10000, alpha: float = 1.0e-2,
            verbose: bool = True) -> LinearRegressionResult:
        self._cost_history = []
        self._parameter_history = []
        
        for i in range(iterations):
            dj_dw, dj_db = self.compute_gradient(x_train, y_train, self.w, self.b)

            self.b = self.b - alpha * dj_db
            self.w = self.w - alpha * dj_dw

            if i < 100000:
                self._cost_history.append(self.compute_cost(x_train, y_train, self.w, self.b))
                self._parameter_history.append([self.w, self.b])

            if verbose and i % (iterations // 10) == 0:
                print(f"Iteration {i:4}: Cost {self._cost_history[-1]:0.2e} ",
                      f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e} ",
                      f"w: {self.w: 0.3e}, b:{self.b: 0.5e}")

        return LinearRegressionResult(
            w=self.w,
            b=self.b,
            cost_history=self._cost_history,
            parameter_history=self._parameter_history
        ) 