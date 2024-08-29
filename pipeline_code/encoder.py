from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from scipy import sparse
import pandas as pd
from typing import Optional, Dict, List

class DynamicColumnEncoder(BaseEstimator, TransformerMixin):
    """
    Un codificatore personalizzato per le colonne categoriche in un DataFrame. 
    Supporta la codifica One-Hot e la codifica Ordinale per colonne specificate.

    Attributi:
        encoder (OneHotEncoder): Un codificatore One-Hot per colonne categoriche non ordinali.
        ordinal_columns (Optional[Dict[str, List[str]]]): Un dizionario che mappa i nomi delle colonne categoriche ordinale a una lista di categorie ordinate.
        ordinal_encoders (Dict[str, OrdinalEncoder]): Un dizionario che mappa i nomi delle colonne a un oggetto OrdinalEncoder specifico per quella colonna.
        categorical_features (List[str]): Lista di tutte le colonne categoriche individuate.
        ordered_categorical_features (List[str]): Lista delle colonne categoriche che richiedono codifica ordinale.
        unordered_categorical_features (List[str]): Lista delle colonne categoriche che richiedono codifica One-Hot.
    """
    
    def __init__(self, ordinal_columns: Optional[Dict[str, List[str]]] = None):
        """
        Inizializza l'oggetto DynamicColumnEncoder.

        Args:
            ordinal_columns (Optional[Dict[str, List[str]]]): Un dizionario opzionale che mappa i nomi delle colonne categoriche ordinale ad una lista di categorie ordinate.
        """
        self.encoder = OneHotEncoder(sparse_output=True)
        self.ordinal_columns = ordinal_columns
        self.ordinal_encoders: Dict[str, OrdinalEncoder] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DynamicColumnEncoder':
        """
        Adatta il codificatore ai dati di input.

        Args:
            X (pd.DataFrame): Il DataFrame di input contenente le colonne categoriche da codificare.
            y (Optional[pd.Series]): Variabile target (non utilizzata in questo trasformatore).

        Returns:
            DynamicColumnEncoder: L'oggetto adattato.
        """
        self.categorical_features = [col for col in X.columns if "cat__" in col]
        
        if self.ordinal_columns:
            self.ordered_categorical_features = [col for col in self.categorical_features if col in self.ordinal_columns.keys()]
            self.unordered_categorical_features = [col for col in self.categorical_features if col not in self.ordinal_columns.keys()]

            if self.unordered_categorical_features:
                self.encoder.fit(X[self.unordered_categorical_features])
            
            for col in self.ordered_categorical_features:
                categories = self.ordinal_columns[col]
                self.ordinal_encoders[col] = OrdinalEncoder(categories=[categories])
                try: 
                    self.ordinal_encoders[col].fit(X[[col]].astype('float64').astype('category'))
                except ValueError:
                    self.ordinal_encoders[col].fit(X[[col]].astype('category'))
        else:
            self.encoder.fit(X[self.categorical_features])
                
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        """
        Trasforma il DataFrame di input codificando le colonne categoriche.

        Args:
            X (pd.DataFrame): Il DataFrame di input da trasformare.

        Returns:
            sparse.csr_matrix: Una matrice sparsa CSR che rappresenta il DataFrame codificato.
        """
        X_encoded: Optional[sparse.csr_matrix] = None
        X_ordinal: Optional[sparse.csr_matrix] = None

        if self.ordinal_columns:
            if self.unordered_categorical_features:
                X_encoded = self.encoder.transform(X[self.unordered_categorical_features].astype('category'))
            
            if self.ordered_categorical_features:
                result = None
                for col in self.ordered_categorical_features:
                    try:
                        result = sparse.hstack([result, self.ordinal_encoders[col].transform(X[[col]].astype('float64').astype('category'))])
                    except ValueError:
                        result = sparse.hstack([result, self.ordinal_encoders[col].fit_transform(X[[col]].astype('category'))])
                X_ordinal = sparse.hstack([sparse.csr_matrix(result)])
        
        else:
            X_encoded = self.encoder.transform(X[self.categorical_features].astype('category'))

        X_remaining = sparse.csr_matrix(X.drop(columns=self.categorical_features, errors='ignore').astype(float))
        
        arrays_to_concatenate = [X_remaining]
        if X_ordinal is not None:
            arrays_to_concatenate.append(X_ordinal)
        if X_encoded is not None:
            arrays_to_concatenate.append(X_encoded)

        return sparse.hstack(arrays_to_concatenate)