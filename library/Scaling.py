# -*- coding: utf-8 -*-
"""
Scaling

@author: Charlotte Meinke, Silvan Hornstein, Rebecca Delfendahl, Till Adam & Kevin Hilbert
"""

# %% Import Packages
import re
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# %% Scaling
# From here: https://stackoverflow.com/questions/72572232/how-to-preserve-column-order-after-applying-sklearn-compose-columntransformer-on?noredirect=1&lq=1


class ReorderColumnTransformer(BaseEstimator, TransformerMixin):
    index_pattern = re.compile(r'\d+$')

    def __init__(self, column_transformer):
        self.column_transformer = column_transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        order_after_column_transform = [int(self.index_pattern.search(
            col).group()) for col in self.column_transformer.get_feature_names_out()]
        order_inverse = np.zeros(len(order_after_column_transform), dtype=int)
        order_inverse[order_after_column_transform] = np.arange(
            len(order_after_column_transform))
        return X[:, order_inverse]


class ZScalerDimVars(BaseEstimator, TransformerMixin):
    """
    ZScalerDimVars is a custom transformer that standardizes dimensional variables (features with more than two unique values)
    while keeping the remaining variables and the variable indices unchanged.

    Parameters:
    None

    Attributes:
    - scaler : sklearn.pipeline.Pipeline
      A pipeline containing a ColumnTransformer for standardizing dimensional variables and a ReorderColumnTransformer
      to maintain the order of variables.
    - dim_features : array-like, shape (n_features,)
      A boolean array indicating which features are dimensional (True) or not (False).

    Methods:
    - fit(X, y=None)
      Fit the ZScalerDimVars on the input data.

    - transform(X)
      Transform the input data by standardizing dimensional variables and keeping non-dimensional variables unchanged.

    - fit_transform(X, y=None)
      Fit and transform the input data in a single step.

    Raises:
    - ValueError
      If the transformer is used before fitting. Call fit method before transform.
    """

    def __init__(self):
        self.scaler = None
        self.dim_features = None

    def fit(self, X, y=None):
        unique_counts = (np.sort(X, axis=0)[
                         :-1] != np.sort(X, axis=0)[1:]).sum(axis=0) + 1
        self.dim_features = unique_counts > 2
        dim_col_transformer = ColumnTransformer(transformers=[('standard', StandardScaler(), self.dim_features)],
                                                remainder='passthrough')
        self.scaler = make_pipeline(
            dim_col_transformer, ReorderColumnTransformer(column_transformer=dim_col_transformer))
        self.scaler.fit(X)

    def transform(self, X):
        if self.scaler is None or self.dim_features is None:
            raise ValueError(
                "Scaler not fitted. Call fit method before transform.")
        return self.scaler.transform(X)

    def fit_transform(self, X, y=None):
        if self.scaler is None or self.dim_features is None:
            self.fit(X)
        return self.transform(X)
