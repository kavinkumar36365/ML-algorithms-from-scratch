import numpy as np

def sigmoid(linear_prediction):
    prediction = 1/(1+ np.exp(-linear_prediction))
    return prediction


class LogisticRegression:
    def __init__(self,lr=0.01,n_iters=1000):
        self.learning_rate=lr
        self.no_of_iterations=n_iters
        self.weights=None   #initializing weights and bias
        self.bias=None

    def fit(self,X,Y):
        no_of_samples,no_of_features=X.shape
        self.weights=np.zeros(no_of_features)
        self.bias=0

        for i in range(self.no_of_iterations):
            linear_prediction = np.dot(self.weights, X.T) + self.bias
            y_predict = sigmoid(linear_prediction)

            dw = (1/no_of_samples)*np.dot(X.T,(y_predict-Y))   #computing gradients
            db = (1/no_of_samples)*np.sum(y_predict-Y)

            self.weights = self.weights - self.learning_rate * dw #updating weights and bias
            self.bias = self.bias - self.learning_rate * db


    def predict(self,X):
        linear_prediction = np.dot(self.weights,X.T)+self.bias
        prediction = sigmoid(linear_prediction)
        class_prediction = [0 if y<=0.5 else 1 for y in prediction]
        return class_prediction