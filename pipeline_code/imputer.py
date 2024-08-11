from sklearn.impute import SimpleImputer, KNNImputer

class CustomKNNImputer(KNNImputer):
    def __init__(self, n_neighbors = 2, weights = "uniform", train_labels = None, **kwargs):
        super().__init__(n_neighbors=n_neighbors, weights=weights, **kwargs)
        self.train_labels = train_labels
        self.imputer_by_class_ = {}
        
        if self.train_labels is None:
            self.main_imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)

    def fit(self, X, y = None):
        if self.train_labels is not None:
            for cls in self.train_labels.unique():
                indices = self.train_labels[self.train_labels.index.isin(X.index) & (self.train_labels == cls)].index
                self.imputer_by_class_[cls] = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights).fit(X.loc[indices])
        else:
            self.main_imputer.fit(X)
        return self

    def transform(self, X):
        if self.train_labels is not None:
            for cls, imputer in self.imputer_by_class_.items():
                indices = self.train_labels[self.train_labels.index.isin(X.index) & (self.train_labels == cls)].index
                X.loc[indices] = imputer.transform(X.loc[indices])
        else:
            X = self.main_imputer.transform(X)
        return X

class CustomSimpleImputer(SimpleImputer):
    def __init__(self, strategy = 'most_frequent', train_labels = None, **kwargs):
        super().__init__(strategy=strategy, **kwargs)
        self.train_labels = train_labels
        self.imputer_by_class_ = {}
        
        if self.train_labels is None:
            self.main_imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y = None):
        if self.train_labels is not None:
            for cls in self.train_labels.unique():
                indices = self.train_labels[self.train_labels.index.isin(X.index) & (self.train_labels == cls)].index
                self.imputer_by_class_[cls] = SimpleImputer(strategy=self.strategy).fit(X.loc[indices])
        else:
            self.main_imputer.fit(X)
        return self

    def transform(self, X):
        if self.train_labels is not None:
            for cls, imputer in self.imputer_by_class_.items():
                indices = self.train_labels[self.train_labels.index.isin(X.index) & (self.train_labels == cls)].index
                X.loc[indices] = imputer.transform(X.loc[indices])
        else:
            X = self.main_imputer.transform(X)
        return X