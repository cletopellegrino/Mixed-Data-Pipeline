import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression 
from sklearn.metrics import mutual_info_score
from sklearn.utils.validation import check_is_fitted
from scipy.stats import gaussian_kde

EPSILON = 1e-10  # A small positive value to avoid division by zero

def diff_entropy(data_cont):
    density = gaussian_kde(data_cont)
    x_space = np.linspace(min(data_cont), max(data_cont), 1000)
    pdf = density.evaluate(x_space)
    return -np.trapz(pdf * np.log2(pdf + EPSILON), x=x_space)

def entropy_discrete(data_disc):
    counts = Counter(data_disc)
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values()]
    return -sum(p * np.log2(p + EPSILON) for p in probabilities if p > 0)

class MutualInfoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, len_numerical = 2, k = 'all'):
        self.k = k
        self.len_numerical = len_numerical

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        for col in X.columns[:self.len_numerical]:
            X[col] = X[col].astype('float64')
        for col in X.columns[self.len_numerical:]:
            X[col] = X[col].astype('category')
        
        self.feature_names_ = X.columns
        self.num_features_ = X.shape[1]
    
        self.mi_with_class_ = []
    
        for i in range(self.num_features_):
            feature = X.iloc[:, i]
    
            if str(feature.dtype) == "category":
                # Feature is categorical, target is categorical
                self.mi_with_class_.append(mutual_info_score(feature.values, y))
            else:
                # Feature is numerical, target is categorical
                self.mi_with_class_.append(mutual_info_classif(feature.values.reshape(-1, 1), y, random_state=0)[0])
    
        self.selected_features_ = self._select_features(X, y)
    
        return self
    
    def _select_features(self, X, y):
        if self.k == 'all':
            return self.feature_names_.tolist()
    
        selected_features = []
        num_selected = 1
        
        current_best = np.argmax(self.mi_with_class_)
        selected_features.append(self.feature_names_[current_best])
    
        while num_selected < self.k:
            best_feature_idx = -1
            best_G = -np.inf
            
            for candidate_idx in range(self.num_features_):
                if self.feature_names_[candidate_idx] not in selected_features:
                
                    mi_candidate_class = self.mi_with_class_[candidate_idx]
                    
                    # Compute NMI(candidate_feature, already selected features)
                    nmi_sum = 0
                    for selected_feature in selected_features:
                        feature_x = X.iloc[:, candidate_idx]
                        feature_y = X[selected_feature]
                        
                        if str(feature_x.dtype) == "float64" and str(feature_y.dtype) == "float64":
                            H_x = diff_entropy(feature_x.values)
                            H_y = diff_entropy(feature_y.values)
                            nmi_sum += mutual_info_regression(feature_x.values.reshape(-1, 1), feature_y.values, random_state=0) / min(H_x, H_y)
                        
                        elif str(feature_x.dtype) == "category" and str(feature_y.dtype) == "category":
                            H_x = entropy_discrete(feature_x.values)
                            H_y = entropy_discrete(feature_y.values)
                            nmi_sum += mutual_info_score(feature_x.values, feature_y.values) / min(H_x, H_y)
                        
                        elif str(feature_x.dtype) == "category" and str(feature_y.dtype) == "float64":
                            H_x = entropy_discrete(feature_x.values)
                            H_y = diff_entropy(feature_y.values)
                            nmi_sum += mutual_info_classif(feature_y.values.reshape(-1, 1), feature_x.values, random_state=0) / min(H_x, H_y)
                        
                        elif str(feature_x.dtype) == "float64" and str(feature_y.dtype) == "category":
                            H_x = diff_entropy(feature_x.values)
                            H_y = entropy_discrete(feature_y.values)
                            nmi_sum += mutual_info_classif(feature_x.values.reshape(-1, 1), feature_y.values, random_state=0) / min(H_x, H_y)
                        
                    S = num_selected
                    G = mi_candidate_class - (nmi_sum / S if S > 0 else 0)
                
                    if G > best_G:
                        best_G = G
                        best_feature_idx = candidate_idx
            
            if best_feature_idx == -1:
                break
    
            selected_features.append(self.feature_names_[best_feature_idx])
            num_selected += 1
            
        return selected_features
    
    def transform(self, X):
        check_is_fitted(self, 'selected_features_')
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for col in X.columns[:self.len_numerical]:
            X[col] = X[col].astype('float64')
        for col in X.columns[self.len_numerical:]:
            X[col] = X[col].astype('category')
        
        for column in X.columns:
            if column not in self.selected_features_:
                X = X.drop(columns=[column])

        return X
    
    def _get_support_mask(self):
        check_is_fitted(self, 'selected_features_')
        mask = np.zeros(self.num_features_, dtype=bool)
        for feature_name in self.selected_features_:
            mask[self.feature_names_.get_loc(feature_name)] = True
        return mask