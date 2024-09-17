# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:08:02 2023

@author: Charlotte Meinke, Silvan Hornstein, Rebecca Delfendahl, Till Adam & Kevin Hilbert
"""
# %% Import packages
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import BayesianRidge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# %% Define Functions and Classes for Imputer
# CAVE: all function are intended to work with dataframes as we need the columns names for one-hot encoding
# The output is a numpy-array!


class BinaryRecoder(BaseEstimator, TransformerMixin):
    """
    Recode binary variables: 0 to -0.5, 1 to 0.5.

    Parameters:
    None

    Methods:
    - fit: No training required.
    - transform: Recode binary variables.
    - get_feature_names_out: Return the input feature names as there is no change.
    """

    def fit(self, X, y=None):
        pass
        return self

    def transform(self, X):
        X[X == 0] = -0.5
        X[X == 1] = 0.5
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class RemoveVarsTooManyMissings(TransformerMixin):
    """
    Remove variables with too many missing values.

    Parameters:
    - threshold: The threshold percentage of missing values to consider for removal.

    Methods:
    - fit: Calculate the percentage of missing values for each column and identify columns to remove.
    - transform: Drop columns with too many missing values.
    - get_support: Return a boolean mask indicating which features are retained.
    - _get_support_mask: Return a boolean mask indicating which features are retained (internal method).
    - get_feature_names_out: Return the new column names after removal of variables with too many missing values.
    """

    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.columns_to_remove = None
        self.new_col_names = None

    def fit(self, X, y=None):
        # Calculate the percentage of missing values for each column
        missing_percentages = X.isna().mean()
        # Identify columns that exceed the threshold
        self.columns_to_remove = missing_percentages[missing_percentages >
                                                     self.threshold].index
        return self

    def transform(self, X):
        # Drop columns with too many missing values
        X = X.drop(columns=self.columns_to_remove)
        self.new_col_names = X.columns
        return X

    def _get_support_mask(self):
        return self.missing_percentages > self.threshold

    def get_support(self):
        # Return a boolean mask indicating which features are retained
        return ~self._get_support_mask()

    def get_feature_names_out(self, input_features=None):
        return self.new_col_names


def initialize_preprocessor(X,cat_vars):
    """
    Initialize a preprocessor pipeline for handling missing values and encoding categorical variables.
    # If categorical variables are not one-hot-encoded yet, we have 3 types of data:
    # binary, categorical, dimensional
    # We need to deal with each type of data differently. 
    # Binary variables: Imputing (Recoding has already taken place before)
    # Categorical variables: Imputing, Onehot-Encoding, Recoding
    # Dimensional variables: Imputing AFTER one-hot encoding of categorical variables

    Parameters:
    - X: Input dataframe.

    Returns:
    - preprocessor: Preprocessing pipeline.

    The preprocessor pipeline handles:
    - Removal of variables with too many missing values.
    - Imputation and encoding of binary and categorical variables.
    - Imputation of dimensional variables after one-hot encoding of categorical variables.
    """
    # Dimensional preprocessor for iterative imputation
    dim_preprocessor = IterativeImputer(estimator=BayesianRidge(),
                                        sample_posterior=True, max_iter=10,
                                        initial_strategy="mean", random_state=0)

    # Get feature names of binary and categorical variables
    # Define bin vars (Variables that contain only -0.5 and 0.5)
    binary_vars = [
        col for col in X.columns
        if np.all(np.isin(np.unique(X[col].dropna()), [-0.5, 0.5], assume_unique=True))
    ]

    # Define categorical variables (Variables that contain strings)
    if not cat_vars:
        is_string = np.vectorize(lambda x: isinstance(x, str))
        cat_vars = [col for col in X.columns if any(is_string(X[col]))]

    # Binary transformer pipeline
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    # Categorical transformer pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop="first", handle_unknown="ignore")),
        ("recode", BinaryRecoder())
    ])
    # Column transformer for processing binary and categorical variables
    processor_nondims = ColumnTransformer(transformers=[
        ('bin', binary_transformer, binary_vars),
        ('cat', categorical_transformer, cat_vars)],
        remainder="passthrough")

    # Full pipeline including imputation and MICE
    Preprocessor = Pipeline(steps=[
        ("Remove_vars_too_many_miss", RemoveVarsTooManyMissings()),
        ("Impute_and_encode_nondims", processor_nondims),
        ('Impute_dims', dim_preprocessor)
    ])
    return Preprocessor

# %% Final imputer class


class MiceModeImputer_pipe(BaseEstimator, TransformerMixin):
    """
    Impute missing values in an array by removing variables with too many missing values
    and imputing missing values in remaining variables.

    Methods:
    - return_feat_names: Remove prefix from feature names (if present).
    - fit: Initialize the preprocessor pipeline, fit it on the data, and extract feature names.
    - transform: Apply the preprocessor on new data.
    - fit_transform: Initialize the preprocessor pipeline, fit and transform the data, and extract feature names.
    """

    def __init__(self):
        self.new_feat_names = None

    def return_feat_names(self, feat_names_a):
        remove_prefix = np.vectorize(lambda x: x.split('__')[
                                     1] if '__' in x else x)
        new_feat_names = remove_prefix(feat_names_a)

        return new_feat_names

    def fit(self, X, cat, y=None):
        # Initialize the preprocessor pipeline
        self.Preprocessor = initialize_preprocessor(X,cat)

        # Fit the preprocessor on the data
        self.Preprocessor.fit(X)

        # Extract and clean feature names
        feat_names_a = self.Preprocessor.get_feature_names_out()
        self.new_feat_names = self.return_feat_names(feat_names_a)

        return self

    def transform(self, X):
        # Apply the preprocessor on new data
        X_transformed = self.Preprocessor.transform(X)
        return X_transformed

    def fit_transform(self, X, cat=None):
        self.Preprocessor = initialize_preprocessor(X,cat)
        # Fit and transform the data using the preprocessor
        X_transformed = self.Preprocessor.fit_transform(X)

        feat_names_a = self.Preprocessor.get_feature_names_out()
        self.new_feat_names = self.return_feat_names(feat_names_a)

        return X_transformed
