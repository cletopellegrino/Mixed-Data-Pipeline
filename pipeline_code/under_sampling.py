import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from model import MixedKNN
from typing import List

class MixedEditedNearestNeighbors(BaseEstimator):
    """
    Un'implementazione personalizzata dell'algoritmo Edited Nearest Neighbors che gestisce sia le variabili numeriche che quelle categoriali.
    """

    def __init__(self, k_neighbors: int = 3, sampling_strategy: any = None) -> None:
        """
        Inizializza l'oggetto MixedEditedNearestNeighbors.

        Args:
            k_neighbors (int): Il numero di vicini più vicini da considerare.
            sampling_strategy (any): La strategia di campionamento.
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.categorical_features = []
        self.continuous_features = []

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Adatta il modello ai dati e risampla rimuovendo i punti di dati incoerenti.

        Args:
            X (np.ndarray): I dati di input.
            y (np.ndarray): Le etichette di output.

        Returns:
            (np.ndarray, np.ndarray): I dati risampolati e le etichette risampolate.
        """
        self.categorical_features = [col for col in X.columns if "cat__" in col]
        self.continuous_features = [col for col in X.columns if "num__" in col]

        X, y = check_X_y(X, y)
        knn = MixedKNN(n_neighbors=self.k_neighbors)
        knn.fit(X, y)
        
        while True:
            indices_to_remove: List[int] = []

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

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.continuous_features + self.categorical_features)
        return X, y