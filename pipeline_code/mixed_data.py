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
from under_sampling import MixedEditedNearestNeighbors
from model import MixedDecisionTree, MixedGaussianNB, MixedKNN
import pandas as pd

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
    model: str = 'MixedRandomForest',
):

    pipeline_steps = []

    if len(numerical_features) > 0:
        numerical_pipeline = Pipeline([('imputer', CustomKNNImputer(train_labels=target))]) if imputers_needed else 'passthrough'
        if scaler is not None:
            numerical_pipeline = Pipeline([('imputer', CustomKNNImputer(train_labels=target)), ('scaler', scaler)]) if imputers_needed else Pipeline([('scaler', scaler)])
        pipeline_steps.append(('num', numerical_pipeline, numerical_features))

    if len(categorical_features) > 0:
        categorical_unord_pipeline = Pipeline([('imputer', CustomSimpleImputer(train_labels=target))]) if imputers_needed else 'passthrough'
        pipeline_steps.append(('cat', categorical_unord_pipeline, categorical_features))

    preprocessing = ColumnTransformer(pipeline_steps)

    pipeline_steps = [
        ('preprocessing', preprocessing),
        ('feature_selector', MutualInfoFeatureSelector(len_numerical=len(numerical_features)))
    ]

    if oversample_needed:
        pipeline_steps.append(('oversampler', SMOTE_ENC(sampling_strategy=oversampling_strategy, categorical_features=categorical_features, continuous_features=numerical_features)))
        
    if undersample_needed:
        pipeline_steps.append(('undersampler', MixedEditedNearestNeighbors(sampling_strategy=undersampling_strategy)))

    if model == "MixedRandomForest":
        pipeline_steps.append(('model', BaggingClassifier(random_state=0, n_jobs=-1, n_estimators=30, estimator=MixedDecisionTree())))
    elif model == "AdaBoostMixedGaussianNB":
        pipeline_steps.append(('model', AdaBoostClassifier(random_state=0, estimator=MixedGaussianNB(), algorithm="SAMME")))
    elif model == "MixedKNN":
        pipeline_steps.append(('model', MixedKNN()))
    else:
        raise ValueError("Model not supported")
    

    return Pipeline(pipeline_steps)

if __name__ == '__main__':
    pass