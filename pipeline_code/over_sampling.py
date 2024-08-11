from collections import Counter
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

class SMOTE_ENC(BaseEstimator):
    def __init__(self, sampling_strategy, k_neighbors=5, borderline=False, categorical_features=None, continuous_features=None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.borderline = borderline
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

    def fit_resample(self, X, y):
        if isinstance(self.sampling_strategy, dict):
            sampling_strategy_dict = {}
            self.class_counts = dict(Counter(y))
            for cls, value in self.sampling_strategy.items():
                if value:
                    for false_cls, false_value in self.sampling_strategy.items():
                        if not false_value:
                            sampling_strategy_dict[cls] = self.class_counts[false_cls]
                else:
                    sampling_strategy_dict[cls] = self.class_counts[cls]
            self.sampling_strategy = sampling_strategy_dict

        features_dict = {i: col for i, col in enumerate(self.continuous_features + self.categorical_features)}
        current_cols = []

        for i, col in features_dict.items():
            if i in X.columns:
                current_cols.append(col)
        
        self.categorical_features = [col for col in self.categorical_features if col in current_cols]
        self.continuous_features = [col for col in self.continuous_features if col in current_cols]

        X = X.rename(columns=features_dict)
        y = pd.Series(y).reset_index(drop=True)
        
        minority_class = y.value_counts().idxmin()
        minority_indices = np.where(y == minority_class)[0]
        t = len(minority_indices)
        s = len(y)
        ir = t / s
        
        if len(self.continuous_features) > 0:
            std_devs = X[self.continuous_features].std()
            m = std_devs.median()
        else:
            m = 0

        encoded_values = {}
        for feature in self.categorical_features:
            encoded_values[feature] = {}
            labels = X[feature].unique()
            for label in labels:
                e = sum(X[feature] == label)
                o = sum((X[feature] == label) & (y == minority_class))
                expected_minority_label_count = e * ir
                if expected_minority_label_count != 0:
                    chi = (o - expected_minority_label_count) / expected_minority_label_count
                else:
                    chi = 0  # Handle the zero case
                if len(self.continuous_features) > 0:
                    l = chi * m
                else:
                    l = chi
                encoded_values[feature][label] = l
                # Temporarily convert the categorical column to object type
                X[feature] = X[feature].astype(object)
                X.loc[(X[feature] == label), feature] = l

        if self.borderline:
            smote = BorderlineSMOTE(random_state=0, sampling_strategy=self.sampling_strategy, k_neighbors=self.k_neighbors)
        else:
            smote = SMOTE(random_state=0, sampling_strategy=self.sampling_strategy, k_neighbors=self.k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        for feature in self.categorical_features:
            reverse_encoding = {v: k for k, v in encoded_values[feature].items()}
            X_resampled[feature] = X_resampled[feature].apply(lambda x: reverse_encoding.get(x, x))
            # Convert back to categorical type
            X_resampled[feature] = X_resampled[feature].astype("category")
        
        # Fix categorical values for synthetic samples
        synthetic_indices = range(len(y), len(y_resampled))

        for feature in self.categorical_features:
            if len(self.continuous_features) > 0:
                original_values_string = np.array(list(encoded_values[feature].keys()))
                X_resampled[feature] = X_resampled[feature].astype(object)

                nn = NearestNeighbors(n_neighbors=self.k_neighbors)
                nn.fit(X_resampled.iloc[:len(y)][self.continuous_features].values)
                
                for idx in synthetic_indices:
                    current_value = X_resampled.at[idx, feature]
                    # Controlla se il valore corrente è già convertito
                    if current_value in original_values_string:
                        continue

                    neighbors = nn.kneighbors([X_resampled.iloc[idx][self.continuous_features]], return_distance=False)[0]
                    majority_value = X_resampled.iloc[neighbors][feature].mode()[0]
                    X_resampled.at[idx, feature] = majority_value
            else:
                original_values_dict = encoded_values[feature]
                original_values_string = np.array(list(encoded_values[feature].keys()))
                original_values_val = np.array(list(encoded_values[feature].values()))
                
                X_resampled[feature] = X_resampled[feature].astype(object)
                
                for idx in synthetic_indices:
                    current_value = X_resampled.at[idx, feature]
                    # Controlla se il valore corrente è già convertito
                    if current_value in original_values_string:
                        continue
                    # Trova il valore più vicino tra quelli originali
                    difference = np.abs(original_values_val - current_value)
                    nearest_value = original_values_val[np.argmin(difference)]
                    for string, val in original_values_dict.items():
                        if val == nearest_value:
                            nearest_string = string
                            break
                    
                    X_resampled.at[idx, feature] = nearest_string

                reverse_encoding = {v: k for k, v in encoded_values[feature].items()}
                X_resampled[feature] = X_resampled[feature].apply(lambda x: reverse_encoding.get(x, x))

            # Convert back to categorical type
            X_resampled[feature] = X_resampled[feature].astype("category")

        return X_resampled, y_resampled