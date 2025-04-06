import numpy as np
from linear_regression.src import LinearRegressionModel, plot_regression_results

def main():
    # Generate some random training data
    x_train = np.random.uniform(0, 10, size=100)
    y_train = np.random.uniform(0, 1000, size=100)
    
    # Create and train the model
    model = LinearRegressionModel()
    result = model.gradient_descent(x_train, y_train, iterations=10000, alpha=1.0e-2)
    
    # Make a prediction for a new value
    x_pred = 5.0
    y_pred = result.predict(x_pred)
    print(f"Predicted y for x={x_pred}: {y_pred}")
    
    # Plot the results
    plot_regression_results(x_train, y_train, result, x_pred)

if __name__ == "__main__":
    main() 