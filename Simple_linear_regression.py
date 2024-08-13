import numpy as np

class Simple_liner_regression:
    def __init__(self):
        self.m = None
        self.b = None

    def best_slope_and_intercept(self, x, y):
        # Calculating the slope (m) and intercept (b)
        self.m = ((np.mean(x) * np.mean(y)) - np.mean(x * y)) / (np.mean(x) * np.mean(x) - np.mean(x * x))
        self.b = np.mean(y) - self.m * np.mean(x)

    def predict_y(self, x):
        # Predicting y values using the calculated slope and intercept
        return (self.m * x) + self.b
