import numpy as np
from sklearn.utils import shuffle


# ------------------------------
# Perceptron Class (Binary)
# ------------------------------
class PerceptronBinary:
    def __init__(self, input_dim, learning_rate=0.01, n_epochs=100, mse_threshold=0.01, add_bias=True):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.mse_threshold = mse_threshold
        self.add_bias = add_bias
        self.W = np.zeros(input_dim)
        self.errors_ = []

    def train(self, X, y):
        for epoch in range(self.n_epochs):
            X, y = shuffle(X, y)
            mse = 0
            for xi, target in zip(X, y):
                y_pred = self.predict_single(xi)
                error = target - y_pred
                self.W += self.lr * error * xi
                mse += error ** 2
            mse /= len(X)
            self.errors_.append(mse)
            if mse < self.mse_threshold:
                print(f"✅ Early stopping at epoch {epoch+1} (MSE={mse:.4f})")
                break

    def predict_single(self, x):
        return 1 if np.dot(self.W, x) >= 0 else 0

    def predict(self, X):
        return np.array([self.predict_single(xi) for xi in X])
