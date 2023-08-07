import numpy as np

class LinearRegression:

    def __init__(self,lr=0.01,n_iters=1000):
        self.learning_rate=lr
        self.no_of_iterations=n_iters
        self.weights=None
        self.bias=None

    def fit(self, X, y):
        no_of_samples,no_of_features=X.shape
        self.weights=np.zeros(no_of_features) #initializing weights and bias
        self.bias=0


        for i in range(self.no_of_iterations):
            y_predict=np.dot(self.weights,X.T) + self.bias
            dw = (1/no_of_samples)*(np.dot(X.T, (y_predict - y))) # computing derivatives of loss function
            db = (1/no_of_samples)*np.sum(y_predict - y)
            # updation of parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db


    def predict(self,X):
        y_predict=np.dot(self.weights,X.T) + self.bias
        return y_predict