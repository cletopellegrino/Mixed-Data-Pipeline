from typing import List, Optional, Union, Dict
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from feature_selection import MutualInfoFeatureSelector
from imputer import CustomKNNImputer, CustomSimpleImputer
from over_sampling import SMOTE_ENC
from encoder import DynamicColumnEncoder
from under_sampling import MixedEditedNearestNeighbors
from model import MixedDecisionTree, MixedGaussianNB, MixedKNN
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor

def create_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    imputers_needed: bool = False,
    scaler: Optional[BaseEstimator] = StandardScaler(),
    oversample_needed: bool = False,
    oversampling_strategy: Optional[Union[str, float, Dict[str, bool]]] = 'minority',
    undersample_needed: bool = False,
    undersampling_strategy: Optional[Union[str, float, List[str], Dict[str, float]]] = 'majority',
    target: Optional[pd.Series] = None,
    model: str = 'RandomForestClf',
):

    pipeline_steps = []

    if len(numerical_features) > 0:
        numerical_pipeline = Pipeline([('imputer', CustomKNNImputer(train_labels=target))]) if imputers_needed else 'passthrough'
        if scaler is not None:
            numerical_pipeline = Pipeline([('imputer', CustomKNNImputer(train_labels=target)), ('scaler', scaler)]) if imputers_needed else Pipeline([('scaler', scaler)])
        pipeline_steps.append(('num', numerical_pipeline, numerical_features))

    if len(categorical_features) > 0:
        categorical_pipeline = Pipeline([('imputer', CustomSimpleImputer(train_labels=target))]) if imputers_needed else 'passthrough'
        pipeline_steps.append(('cat', categorical_pipeline, categorical_features))

    preprocessing = ColumnTransformer(pipeline_steps)
    preprocessing.set_output(transform="pandas") # per eliminare il warning

    pipeline_steps = [
        ('preprocessing', preprocessing),
        ('feature_selector', MutualInfoFeatureSelector())
    ]

    if oversample_needed:
        pipeline_steps.append(('oversampler', SMOTE_ENC(sampling_strategy=oversampling_strategy)))
        
    if undersample_needed:
        pipeline_steps.append(('undersampler', MixedEditedNearestNeighbors(sampling_strategy=undersampling_strategy)))

    match model:
        case "RandomForestClf":
            if categorical_features:
                pipeline_steps.append(('model', BaggingClassifier(random_state=0, n_jobs=-1, n_estimators=30, estimator=MixedDecisionTree())))
            else:
                pipeline_steps.append(('model', RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=30)))
        
        case "RandomForestReg":
            if categorical_features:
                pipeline_steps.append(('encoder', DynamicColumnEncoder()))

            pipeline_steps.append(('model', RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=30)))
        
        case "DecisionTreeClf":
            if categorical_features:
                pipeline_steps.append(('model', MixedDecisionTree()))
            else:
                pipeline_steps.append(('model', DecisionTreeClassifier(random_state=0)))
        
        case "DecisionTreeReg":
            if categorical_features:
                pipeline_steps.append(('encoder', DynamicColumnEncoder()))

            pipeline_steps.append(('model', DecisionTreeRegressor(random_state=0)))
        
        case "AdaBoostGaussianNB":
            if categorical_features:
                pipeline_steps.append(('model', AdaBoostClassifier(random_state=0, estimator=MixedGaussianNB(), algorithm="SAMME", n_estimators=30)))
            else:
                pipeline_steps.append(('model', AdaBoostClassifier(random_state=0, estimator=GaussianNB(), algorithm="SAMME")))
        
        case "KNNClf":
            if categorical_features:
                pipeline_steps.append(('model', MixedKNN()))
            else:
                pipeline_steps.append(('model', KNeighborsClassifier()))
        
        case "KNNReg":
            if categorical_features:
                pipeline_steps.append(('encoder', DynamicColumnEncoder()))

            pipeline_steps.append(('model', KNeighborsRegressor()))

        case "LogisticRegression":
            if categorical_features:
                pipeline_steps.append(('encoder', DynamicColumnEncoder()))

            pipeline_steps.append(('model', LogisticRegression(random_state=0)))
        
        case "NeuralNetworkClf":
            if categorical_features:
                pipeline_steps.append(('encoder', DynamicColumnEncoder()))
            
            pipeline_steps.append(('model', MLPClassifier(random_state=0)))
        
        case "NeuralNetworkReg":
            if categorical_features:
                pipeline_steps.append(('encoder', DynamicColumnEncoder()))
            
            pipeline_steps.append(('model', MLPRegressor(random_state=0)))
        
        case _:
            raise ValueError(f"Model '{model}' not supported")
    

    return Pipeline(pipeline_steps)

if __name__ == '__main__':
    pass