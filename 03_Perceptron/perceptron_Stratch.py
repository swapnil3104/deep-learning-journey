import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    @staticmethod
    def activation(score):
        return 1 if score >= 0 else 0

    def predict_one(self, x):
        score = np.dot(x, self.weights) + self.bias
        return self.activation(score)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("This implementation requires labels 0 and 1.")

        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = 0.0

        for epoch in range(self.epochs):
            mistakes = 0

            for xi, target in zip(X, y):
                prediction = self.predict_one(xi)
                error = target - prediction

                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error

                if error != 0:
                    mistakes += 1

            print(f"Epoch {epoch + 1}: mistakes = {mistakes}")

            if mistakes == 0:
                print("Training converged.")
                break

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one(x) for x in X])


# AND gate dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

model = Perceptron(learning_rate=0.1, epochs=100)
model.fit(X, y)

print("Weights:", model.weights)
print("Bias:", model.bias)
print("Predictions:", model.predict(X))
print("Targets:    ", y)