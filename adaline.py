# =====================================
# Adaline Model (Final Implementation)
# =====================================
import numpy as np
from sklearn.utils import shuffle

class AdalineBinary:
    def __init__(self, input_dim, learning_rate=0.01, n_epochs=100, mse_threshold=0.01):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.mse_threshold = mse_threshold
        self.W = np.random.randn(input_dim) * 0.01  # small random init
        self.errors_ = []

    def train(self, X, y):
        for epoch in range(self.n_epochs):
            X, y = shuffle(X, y)
            output = X.dot(self.W)
            errors = y - output
            mse = (errors ** 2).mean()

            # Update weights using gradient descent
            self.W += self.lr * X.T.dot(errors) / len(X)

            self.errors_.append(mse)
            print(f"Epoch {epoch+1}/{self.n_epochs} - MSE={mse:.4f}")

            if mse < self.mse_threshold:
                print(f"✅ Converged at epoch {epoch+1} (MSE={mse:.6f})")
                break

    def predict(self, X):
        net_input = X.dot(self.W)
        # Apply threshold at 0
        return np.where(net_input >= 0.0, 1, 0)
