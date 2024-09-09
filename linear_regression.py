import numpy as np

class LinearRegression:
    def __init__(self, l_rate=0.01, iterations = 1000):
        self.l_rate = l_rate
        self.iterations = iterations
        # self._lambda = _lambda
        self.w = None


    def fit(self, x, y, k):  # k represents the degree
        num_samples = x.shape[0]
        self.w = np.random.rand(k+1)
        print(self.w)
        degrees = np.vander(x, N=k+1, increasing=True)
        for i in range(self.iterations):
            predicted_y = np.dot(degrees, self.w)
            e = predicted_y - y 
            gradient = (2 / num_samples) * np.dot(degrees.T, e)
            self.w -= self.l_rate * gradient

    def predict(self, x, k):
        # Create the degree matrix with polynomial features
        degrees = np.vander(x, N=k+1, increasing=True)
        return np.dot(degrees, self.w)  # Compute predicted values

    def MSE(self,y,predicted_y):
        mse = np.mean((y-predicted_y)**2)
        return mse

    def std_deviation(self,y,predicted_y):
        y = np.array(y)
        predicted_y = np.array(predicted_y)
        error = y - predicted_y
        std_dev = np.std(error)
        return std_dev

    def variance(self, y, predicted_y):
        var = self.std_deviation(y, predicted_y) ** 2
        return var