# -*- coding: utf-8 -*-
"""
The script defines customized classes and functions used for preprocessing.

@author: charl
"""

# %% Import functions
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.metrics import pairwise_distances

# %% Selectors


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    FeatureSelector is a two-step procedure for feature exclusion.

    Step 1:
    Features are excluded if:
        - They exhibit no variance.
        - For binary features only, less than 10% of values are in one category.

    Step 2:
    Correlations between dimensional features and Jaccard similarity between binary features are calculated.
    Features are excluded if the correlation or Jaccard similarity is greater than a specified threshold (default: 0.75).
    The decision is based on which of the two features has the largest overall correlation or Jaccard similarity with other features.

    Parameters:
    - cor_threshold : float, optional, default: 0.75
      The correlation and Jaccard similarity threshold for feature exclusion.

    Attributes:
    - is_feat_excluded : array-like, shape (n_features,)
      An array indicating whether each feature is excluded (1) or not (0).
    - background_info : array-like, shape (n_features, 2)
      An array providing additional information about excluded features, where the first column indicates the exclusion reason (1 for Step 1, 2 for correlation, 3 for Jaccard similarity),
      and the second column provides the index of the feature that contributed to the exclusion.

    Methods:
    - fit(X, cor_threshold=0.75, y=None):
      Fit the FeatureSelector on the input data.

    - transform(X):
      Transform the input data by excluding the identified features.

    Returns:
    - self : object
      Returns the instance itself.
    """

    def __init__(self, cor_threshold=0.75):
        self.is_feat_excluded = None
        self.background_info = None
        self.cor_threshold = cor_threshold

    def fit(self, X, y=None):
        X_feat_indices = np.arange(X.shape[1])
        # Throw error if X has missing values
        if np.isnan(X).any():
            raise ValueError(
                "Input array X contains missing values. Remove or impute missings before using this FeatureSelector")

        # Step 1: Exclusion based on variance and least common category
        self.is_feat_excluded = np.zeros(X.shape[1], dtype=int)
        self.background_info = np.full([X.shape[1], 2], np.nan)

        for feat_idx in range(X.shape[1]):
            column = X[:, feat_idx]
            if np.std(column, axis=0) == 0:  # No variance in feature
                self.is_feat_excluded[feat_idx] = 1
            # Less than 10% in one group (For binary features only)
            if np.unique(column).size == 2:
                if np.min(np.unique(column, return_counts=True)[1]) < (len(X) / 10):
                    self.is_feat_excluded[feat_idx] = 1
        self.background_info[:, 0] = self.is_feat_excluded

        # Step 2: Exclusion based on correlations and Jaccard similarity
        # Create seperate dataframes for dimensional and binary variables
        # (sets variables from the "other" type to NA)
        X_dim = X.copy()
        X_bin = X.copy()
        for feat_idx in range(X.shape[1]):
            unique_values = np.unique(X[:, feat_idx])
            if len(unique_values) == 2:  # Binary feature
                X_dim[:, feat_idx] = np.nan
            elif len(unique_values) > 2:  # Categorical feature with more than 2 unique values
                X_bin[:, feat_idx] = np.nan

        # Step 2: Dimensional variables
        while True:
            X_dim_clean = X_dim[:, self.is_feat_excluded == 0]
            X_dim_clean_feat_indices = X_feat_indices[self.is_feat_excluded == 0]

            corr_mat = np.corrcoef(X_dim_clean, rowvar=False)
            # Round corrmat to ensure precision
            corr_mat = np.round(corr_mat, 7)
            np.fill_diagonal(corr_mat, np.nan)

            mean_corr = np.nanmean(np.abs(corr_mat), axis=1)

            # Find the indices of the maximum absolute correlation
            max_corr = np.nanmax(corr_mat)
            max_corr_idx = np.unravel_index(
                np.nanargmax(np.abs(corr_mat)), corr_mat.shape)

            if max_corr > self.cor_threshold:
                if mean_corr[max_corr_idx[0]] > mean_corr[max_corr_idx[1]]:
                    feat_highest_mean_idx = max_corr_idx[0]
                else:
                    feat_highest_mean_idx = max_corr_idx[1]
                feat_highest_mean_idx_in_X = X_dim_clean_feat_indices[feat_highest_mean_idx]
                self.is_feat_excluded[feat_highest_mean_idx_in_X] = 1
                self.background_info[feat_highest_mean_idx_in_X, 0] = 2
                max_corr_idx_array = np.array(max_corr_idx)
                other_feat_idx = max_corr_idx_array[max_corr_idx_array !=
                                                    feat_highest_mean_idx][0]
                other_feat_idx_in_X = X_dim_clean_feat_indices[other_feat_idx]
                self.background_info[feat_highest_mean_idx_in_X,
                                     1] = other_feat_idx_in_X
            else:
                break

        # Step 2: Binary variables
        while True:
            X_bin_clean = X_bin[:, self.is_feat_excluded == 0]
            X_bin_clean_feat_indices = X_feat_indices[self.is_feat_excluded == 0]

            jac_sim_mat = 1 - \
                pairwise_distances(
                    X_bin_clean.T, metric="hamming", force_all_finite="allow-nan")
            # Set jaquard similarity to NA when feature has only NAs
            for feat_idx in range(X_bin_clean.shape[1]):
                column = X_bin_clean[:, feat_idx]
                if len(np.unique(column)) == 0 and np.isnan(np.unique(column)[0]):
                    jac_sim_mat[:, feat_idx] = np.nan
                    jac_sim_mat[feat_idx, :] = np.nan
            np.fill_diagonal(jac_sim_mat, np.nan)  # Fill diagonal with NAs

            mean_jac_sim = np.nanmean(np.abs(jac_sim_mat), axis=1)

            max_sim = np.nanmax(jac_sim_mat)
            max_sim_idx = np.unravel_index(np.nanargmax(
                np.abs(jac_sim_mat)), jac_sim_mat.shape)

            if max_sim > self.cor_threshold:
                if mean_jac_sim[max_sim_idx[0]] > mean_jac_sim[max_sim_idx[1]]:
                    feat_highest_mean_idx = max_sim_idx[0]
                else:
                    feat_highest_mean_idx = max_sim_idx[1]
                feat_highest_mean_idx_in_X = X_bin_clean_feat_indices[feat_highest_mean_idx]
                self.is_feat_excluded[feat_highest_mean_idx_in_X] = 1
                self.background_info[feat_highest_mean_idx_in_X, 0] = 3
                max_sim_idx_array = np.array(max_sim_idx)
                other_feat_idx = max_sim_idx_array[max_sim_idx_array !=
                                                   feat_highest_mean_idx][0]
                other_feat_idx_in_X = X_bin_clean_feat_indices[other_feat_idx]
                self.background_info[feat_highest_mean_idx_in_X,
                                     1] = other_feat_idx_in_X
            else:
                break

        return self

    def transform(self, X):
        X_cleaned = X[:, self.is_feat_excluded == 0]
        return X_cleaned
