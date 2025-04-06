import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .model import LinearRegressionResult

def plot_regression_results(x_train: np.ndarray, y_train: np.ndarray, 
                          result: LinearRegressionResult,
                          x_pred: Optional[float] = None,
                          title: str = "Linear Regression Results") -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x_train, y_train, 'ro', label='Training data')
    plt.plot(x_line, y_line, 'b-', label='Best fit line')
    x_min, x_max = x_train.min(), x_train.max()
    x_line = np.linspace(x_min, x_max, 100)
    y_line = result.predict_batch(x_line)
    plt.plot(x_line, y_line, 'b-', label='Best fit line')
    
    if x_pred is not None:
        y_pred = result.predict(x_pred)
        plt.plot(x_pred, y_pred, 'go', label='Prediction')
        print(f"Predicted y for x={x_pred}: {y_pred}")
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show() 