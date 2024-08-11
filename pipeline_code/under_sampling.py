import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from model import MixedKNN

class MixedEditedNearestNeighbors(BaseEstimator):
    def __init__(self, k_neighbors=3, sampling_strategy=None):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        knn = MixedKNN(n_neighbors=self.k_neighbors)
        knn.fit(X, y)
        
        while True:
            indices_to_remove = []

            for i in range(len(X)):
                if y[i] != self.sampling_strategy:
                    continue

                X_temp = np.delete(X, i, axis=0)
                y_temp = np.delete(y, i)
                knn.fit(X_temp, y_temp)
                y_pred = knn.predict([X[i]])

                if y_pred[0] != y[i]:
                    indices_to_remove.append(i)

            if not indices_to_remove:
                break
            
            X = np.delete(X, indices_to_remove, axis=0)
            y = np.delete(y, indices_to_remove)

            # Refit the model with remaining instances
            knn.fit(X, y)

        return X, y