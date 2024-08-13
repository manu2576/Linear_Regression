import numpy as np

class Multiple_linear_regression:
    def __init__(self, lr=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = lr
        self.epochs = epochs

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.intercept_ = 0
        self.coef_ = np.zeros(n_features)
        for _ in range(self.epochs):
            y_pred = np.dot(X_train, self.coef_) + self.intercept_

            # Gradient for intercept
            der_intercept = -2 * np.mean(y_train - y_pred)
            self.intercept_ -= self.lr * der_intercept

            # Gradient for coefficients
            der_coef = -2 * np.dot(X_train.T, (y_train - y_pred)) / n_samples
            self.coef_ -= self.lr * der_coef

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
