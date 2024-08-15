from sklearn.impute import SimpleImputer, KNNImputer
from pandas import DataFrame, Series
from typing import Optional, Dict

class CustomKNNImputer(KNNImputer):
    """
    Una versione personalizzata dell'imputer KNN.
    """

    def __init__(self, n_neighbors: int = 2, weights: str = "uniform", 
                 train_labels: Optional[Series] = None, **kwargs):
        """
        Inizializza l'imputer.

        Args:
            n_neighbors (int): Il numero di vicini da considerare.
            weights (str): Se 'uniform', pesa uniformemente. Se 'distance', pesa in base alla distanza.
            train_labels (Optional[Series]): Le etichette di addestramento.
            **kwargs: Argomenti aggiuntivi da passare al KNNImputer.
        """
        super().__init__(n_neighbors=n_neighbors, weights=weights, **kwargs)
        self.train_labels = train_labels
        self.imputer_by_class_: Dict = {}
        
        if self.train_labels is None:
            self.main_imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)

    def fit(self, X: DataFrame, y: Optional[Series] = None) -> 'CustomKNNImputer':
        """
        Adatta l'imputer ai dati.

        Args:
            X (DataFrame): I dati di input.
            y (Optional[Series]): Le etichette di output.

        Returns:
            CustomKNNImputer: L'istanza stessa.
        """
        if self.train_labels is not None:
            for cls in self.train_labels.unique():
                indices = self.train_labels[self.train_labels.index.isin(X.index) & (self.train_labels == cls)].index
                self.imputer_by_class_[cls] = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights).fit(X.loc[indices])
        else:
            self.main_imputer.fit(X)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Applica l'imputer ai dati.

        Args:
            X (DataFrame): I dati di input.

        Returns:
            DataFrame: I dati imputati.
        """
        if self.train_labels is not None:
            for cls, imputer in self.imputer_by_class_.items():
                indices = self.train_labels[self.train_labels.index.isin(X.index) & (self.train_labels == cls)].index
                X.loc[indices] = imputer.transform(X.loc[indices])
        else:
            X = self.main_imputer.transform(X)
        return X

class CustomSimpleImputer(SimpleImputer):
    """
    Una versione personalizzata dell'imputer semplice.
    """

    def __init__(self, strategy: str = 'most_frequent', train_labels: Optional[Series] = None, **kwargs):
        """
        Inizializza l'imputer.

        Args:
            strategy (str): La strategia da usare per l'imputazione.
            train_labels (Optional[Series]): Le etichette di addestramento.
            **kwargs: Argomenti aggiuntivi da passare al SimpleImputer.
        """
        super().__init__(strategy=strategy, **kwargs)
        self.train_labels = train_labels
        self.imputer_by_class_: Dict = {}
        
        if self.train_labels is None:
            self.main_imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X: DataFrame, y: Optional[Series] = None) -> 'CustomSimpleImputer':
        """
        Adatta l'imputer ai dati.

        Args:
            X (DataFrame): I dati di input.
            y (Optional[Series]): Le etichette di output.

        Returns:
            CustomSimpleImputer: L'istanza stessa.
        """
        if self.train_labels is not None:
            for cls in self.train_labels.unique():
                indices = self.train_labels[self.train_labels.index.isin(X.index) & (self.train_labels == cls)].index
                self.imputer_by_class_[cls] = SimpleImputer(strategy=self.strategy).fit(X.loc[indices])
        else:
            self.main_imputer.fit(X)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Applica l'imputer ai dati.

        Args:
            X (DataFrame): I dati di input.

        Returns:
            DataFrame: I dati imputati.
        """
        if self.train_labels is not None:
            for cls, imputer in self.imputer_by_class_.items():
                indices = self.train_labels[self.train_labels.index.isin(X.index) & (self.train_labels == cls)].index
                X.loc[indices] = imputer.transform(X.loc[indices])
        else:
            X = self.main_imputer.transform(X)
        return X