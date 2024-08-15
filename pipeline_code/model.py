import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from typing import Optional, List, Dict, Any


class MixedGaussianNB(BaseEstimator, ClassifierMixin):
    """
    Un classificatore Naive Bayes che gestisce sia le variabili numeriche che quelle categoriali.
    """

    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]=None) -> 'MixedGaussianNB':
        """
        Adatta il modello ai dati di addestramento.

        Args:
            X (np.ndarray): I dati di addestramento.
            y (np.ndarray): Le etichette di addestramento.
            sample_weight (Optional[np.ndarray]): I pesi dei campioni.
        """
        X, y = check_X_y(X, y, dtype=None)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)

        self.cat_feature_indices_: List[int] = []
        self.num_feature_indices_: List[int] = []

        temp = X[0]
        for index, i in enumerate(temp):
            if isinstance(i, str):
                self.cat_feature_indices_.append(index)
            else:
                self.num_feature_indices_.append(index)

        self.classes_ = np.unique(y)

        self.theta_ = np.zeros((len(self.classes_), len(self.num_feature_indices_)))
        self.sigma_ = np.zeros((len(self.classes_), len(self.num_feature_indices_)))
        
        for idx, cls in enumerate(self.classes_):
            cls_samples = X[y == cls]
            cls_weights = sample_weight[y == cls]
            if self.num_feature_indices_:
                weighted_mean = np.average(cls_samples[:, self.num_feature_indices_], axis=0, weights=cls_weights)
                weighted_var = np.average((cls_samples[:, self.num_feature_indices_] - weighted_mean) ** 2, axis=0, weights=cls_weights)
                self.theta_[idx, :] = weighted_mean
                self.sigma_[idx, :] = weighted_var
        
        self.cat_prob_: Dict[Any, Dict[Any, float]] = {}
        for idx, cls in enumerate(self.classes_):
            self.cat_prob_[cls] = {}
            cls_samples = X[y == cls]
            cls_weights = sample_weight[y == cls]
            for feature_index in self.cat_feature_indices_:
                values, counts = np.unique(cls_samples[:, feature_index], return_counts=True)
                weighted_counts = np.zeros_like(counts, dtype=np.float64)
                for val_idx, val in enumerate(values):
                    weighted_counts[val_idx] = np.sum(cls_weights[cls_samples[:, feature_index] == val])
                self.cat_prob_[cls][feature_index] = {val: count / np.sum(cls_weights) for val, count in zip(values, weighted_counts)}

        self.class_prior_: Dict[Any, float] = {cls: np.sum(sample_weight[y == cls]) / np.sum(sample_weight) for cls in self.classes_}

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fa previsioni su nuovi dati.

        Args:
            X (np.ndarray): I dati di input.

        Returns:
            np.ndarray: Le previsioni.
        """
        check_is_fitted(self, ["theta_", "sigma_", "cat_prob_", "class_prior_"])
        X = check_array(X, dtype=None)
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calcola la probabilità logaritmica congiunta.

        Args:
            X (np.ndarray): I dati di input.

        Returns:
            np.ndarray: La probabilità logaritmica congiunta.
        """
        jll = []
        for idx, cls in enumerate(self.classes_):
            log_likelihood = np.log(self.class_prior_[cls])
            
            if self.num_feature_indices_:
                mean = self.theta_[idx]
                var = self.sigma_[idx]
                num_likelihood = -0.5 * np.sum(np.log(2. * np.pi * var))
                num_likelihood -= 0.5 * np.sum(((X[:, self.num_feature_indices_] - mean) ** 2) / (var + 1e-9), axis=1)
                log_likelihood += num_likelihood
            
            for feature_index in self.cat_feature_indices_:
                cat_likelihood = np.log([self.cat_prob_[cls][feature_index].get(x, 1e-9) for x in X[:, feature_index]])
                log_likelihood += cat_likelihood
            
            jll.append(log_likelihood)
        
        return np.array(jll).T

class MixedKNN(BaseEstimator, ClassifierMixin):
    """
    Un classificatore KNN che gestisce sia le variabili numeriche che quelle categoriali.
    """

    def __init__(self, n_neighbors: int = 5) -> None:
        """
        Inizializza l'oggetto MixedKNN.

        Args:
            n_neighbors (int): Il numero di vicini più vicini da considerare durante la predizione.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MixedKNN':
        """
        Adatta il modello ai dati di addestramento.

        Args:
            X (np.ndarray): I dati di addestramento.
            y (np.ndarray): Le etichette di addestramento.
        """
        X, y = check_X_y(X, y, dtype=None)
        self.X_ = X
        self.y_ = y

        self.cat_feature_indices_: List[int] = []
        self.num_feature_indices_: List[int] = []

        temp = X[0]
        for index, i in enumerate(temp):
            if isinstance(i, str):
                self.cat_feature_indices_.append(index)
            else:
                self.num_feature_indices_.append(index)

        # Initialize the classes_ attribute
        self.classes_ = np.unique(y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fa previsioni su nuovi dati.

        Args:
            X (np.ndarray): I dati di input.

        Returns:
            np.ndarray: Le previsioni.
        """
        check_is_fitted(self, ["X_", "y_"])
        X = check_array(X, dtype=None)

        predictions = []
        for x in X:
            distances = [self._mixed_distance(x, x_train) for x_train in self.X_]
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_[k_nearest_indices]
            if k_nearest_labels.dtype == 'O':
                # Count the occurrences of each label
                label_counts = {}
                for label in k_nearest_labels:
                    if label in label_counts:
                        label_counts[label] += 1
                    else:
                        label_counts[label] = 1

                # Find the label with the highest count
                most_common_label = max(label_counts, key=label_counts.get)
                predictions.append(most_common_label)
            else:
                predictions.append(np.bincount(k_nearest_labels).argmax())

        return np.array(predictions)

    def _mixed_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calcola la distanza mista tra due campioni.

        Args:
            x (np.ndarray): Primo campione.
            y (np.ndarray): Secondo campione.

        Returns:
            float: La distanza mista.
        """
        num_distance = np.sum((x[self.num_feature_indices_] - y[self.num_feature_indices_]) ** 2)
        cat_distance = np.sum(x[self.cat_feature_indices_] != y[self.cat_feature_indices_])
        return num_distance + cat_distance

class MixedDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Un albero di decisione che gestisce sia le variabili numeriche che quelle categoriali.
    """

    def __init__(self, criterion: str = 'gini', max_depth: Optional[int] = None, min_samples_split: int = 2) -> None:
        """
        Inizializza l'oggetto MixedDecisionTree.

        Args:
            criterion (str): Il criterio da usare per la misura di impurità.
            max_depth (Optional[int]): Il massimo profondità dell'albero.
            min_samples_split (int): Il numero minimo di campioni necessari per dividere un nodo interno.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MixedDecisionTree':
        """
        Adatta il modello ai dati di addestramento.

        Args:
            X (np.ndarray): I dati di addestramento.
            y (np.ndarray): Le etichette di addestramento.
        """
        X, y = self._check_X_y(X, y)

        self.cat_feature_indices_: List[int] = []
        self.num_feature_indices_: List[int] = []

        temp = X[0]
        for index, i in enumerate(temp):
            if isinstance(i, str):
                self.cat_feature_indices_.append(index)
            else:
                self.num_feature_indices_.append(index)

        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fa previsioni su nuovi dati.

        Args:
            X (np.ndarray): I dati di input.

        Returns:
            np.ndarray: Le previsioni.
        """
        X = self._check_array(X)
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def _predict_one(self, x: np.ndarray, tree: dict) -> Any:
        """
        Fa una previsione per un singolo campione.

        Args:
            x (np.ndarray): I dati del campione.
            tree (dict): L'albero di decisione.

        Returns:
            Any: La previsione.
        """
        if tree['is_leaf']:
            return tree['prediction']
        feature_index = tree['feature']
        if feature_index in self.cat_feature_indices_:
            if x[feature_index] in tree['branches']:
                return self._predict_one(x, tree['branches'][x[feature_index]])
            else:
                return tree['prediction']  # Fallback in case of unseen category
        else:
            if x[feature_index] <= tree['threshold']:
                return self._predict_one(x, tree['branches']['left'])
            else:
                return self._predict_one(x, tree['branches']['right'])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        """
        Costruisce l'albero di decisione.

        Args:
            X (np.ndarray): I dati di input.
            y (np.ndarray): Le etichette di output.
            depth (int): La profondità corrente dell'albero.

        Returns:
            dict: L'albero di decisione.
        """
        num_samples, _ = X.shape
        num_labels = len(np.unique(y))

        if depth == self.max_depth or num_samples < self.min_samples_split or num_labels == 1:
            return {'is_leaf': True, 'prediction': self._most_common_label(y)}

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_gain == 0:
            return {'is_leaf': True, 'prediction': self._most_common_label(y)}

        tree = {'is_leaf': False, 'feature': best_feature}

        if best_feature in self.cat_feature_indices_:
            tree['branches'] = {}
            feature_values = np.unique(X[:, best_feature])
            for value in feature_values:
                X_subset, y_subset = X[X[:, best_feature] == value], y[X[:, best_feature] == value]
                tree['branches'][value] = self._build_tree(X_subset, y_subset, depth + 1)
        else:
            left_indices = X[:, best_feature] <= best_threshold
            right_indices = X[:, best_feature] > best_threshold
            tree['threshold'] = best_threshold
            tree['branches'] = {
                'left': self._build_tree(X[left_indices], y[left_indices], depth + 1),
                'right': self._build_tree(X[right_indices], y[right_indices], depth + 1)
            }

        tree['prediction'] = self._most_common_label(y)
        return tree

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Trova la migliore divisione possibile.

        Args:
            X (np.ndarray): I dati di input.
            y (np.ndarray): Le etichette di output.

        Returns:
            tuple: Una tupla contenente il miglior indice di funzione, la migliore soglia e il miglior guadagno.
        """
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature_index in range(X.shape[1]):
            if feature_index in self.cat_feature_indices_:
                feature_values = np.unique(X[:, feature_index])
                for value in feature_values:
                    gain = self._information_gain(y, X[:, feature_index] == value)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_index
                        best_threshold = value
            else:
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:
                    gain = self._information_gain(y, X[:, feature_index] <= threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_index
                        best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _information_gain(self, y: np.ndarray, mask: np.ndarray) -> float:
        """
        Calcola il guadagno di informazione.

        Args:
            y (np.ndarray): Le etichette di output.
            mask (np.ndarray): Una maschera booleana.

        Returns:
            float: Il guadagno di informazione.
        """
        if self.criterion == 'gini':
            return self._gini_gain(y, mask)
        else:
            return self._entropy_gain(y, mask)

    def _gini_gain(self, y: np.ndarray, mask: np.ndarray) -> float:
        """
        Calcola il guadagno di Gini.

        Args:
            y (np.ndarray): Le etichette di output.
            mask (np.ndarray): Una maschera booleana.

        Returns:
            float: Guadagno di Gini.
        """
        p = np.sum(mask) / len(y)
        left_gini = self._gini(y[mask])
        right_gini = self._gini(y[~mask])
        return self._gini(y) - p * left_gini - (1 - p) * right_gini

    def _entropy_gain(self, y: np.ndarray, mask: np.ndarray) -> float:
        """
        Calcola il guadagno di entropia.

        Args:
            y (np.ndarray): Le etichette di output.
            mask (np.ndarray): Una maschera booleana.

        Returns:
            float: Il guadagno di entropia.
        """
        p = np.sum(mask) / len(y)
        left_entropy = self._entropy(y[mask])
        right_entropy = self._entropy(y[~mask])
        return self._entropy(y) - p * left_entropy - (1 - p) * right_entropy

    def _gini(self, y: np.ndarray) -> float:
        """
        Calcola l'indice di Gini.

        Args:
            y (np.ndarray): Le etichette di output.

        Returns:
            float: L'indice di Gini.
        """
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _entropy(self, y: np.ndarray) -> float:
        """
        Calcola l'entropia.

        Args:
            y (np.ndarray): Le etichette di output.

        Returns:
            float: L'entropia.
        """
        m = len(y)
        return -sum((np.sum(y == c) / m) * np.log2(np.sum(y == c) / m) for c in np.unique(y))

    def _most_common_label(self, y: np.ndarray) -> Any:
        """
        Trova l'etichetta più comune.

        Args:
            y (np.ndarray): Le etichette di output.

        Returns:
            Any: L'etichetta più comune.
        """
        return Counter(y).most_common(1)[0][0]

    def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Controlla X e y.

        Args:
            X (np.ndarray): I dati di input.
            y (np.ndarray): Le etichette di output.

        Returns:
            tuple: Una tupla contenente X e y controllati.
        """
        return check_X_y(X, y, dtype=None)

    def _check_array(self, X: np.ndarray) -> np.ndarray:
        """
        Controlla X.

        Args:
            X (np.ndarray): I dati di input.

        Returns:
            np.ndarray: X controllato.
        """
        return check_array(X, dtype=None)