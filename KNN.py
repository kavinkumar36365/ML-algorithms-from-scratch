import numpy as np
from collections import Counter


def euclidean_distance(x1,x2):
    distance=np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self,k):
        self.k=k

    def fit(self,x,y):
        self.x_train=x
        self.y_train=y

    def predict(self,x):
        predictions=[self._predict(i) for i in x]
        return predictions

    def _predict(self,x):

        distances=[euclidean_distance(x,x_train) for x_train in self.x_train]

        k_indices=np.argsort(distances)[:self.k]
        k_nearest_labels=[self.y_train[i] for i in k_indices]

        most_common=Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
