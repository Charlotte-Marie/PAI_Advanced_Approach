# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:59:03 2024

@author: charl & kevin
"""

# %% Import packages

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
# %% Functions


def summarize_features(outcomes, key_feat_names, key_feat_weights):
    """
    Summarize selected features and feature importances across iterations.

    Parameters:
    - outcomes (list): List with one entry per iteration of k-fold cross-validation. 
      Each entry is a dictionary with all information saved per iteration. (results_single_iter)
      This dictionary needs to contain selected features per iteration and their feature weight.
    - key_feat_names (str): Key in the results_single_iter dictionary containing the names of selected features.
    - key_feat_weights (str): Key in the results_single_iter dictionary containing the weights of selected features.

    Returns:
    pd.DataFrame: DataFrame summarizing, for each feature, the selection frequency and mean coefficient across iterations.
    The DataFrame is sorted by selection frequency and mean coefficient in descending order.
    """
    # Create np_array of features that were at least once part of the final model
    feat_names_all = []
    for itera in outcomes:
        feat_names = itera[key_feat_names]
        feat_names_all.append(feat_names)
    feat_names_all = np.concatenate(feat_names_all)
    unique_feat_names = np.unique(feat_names_all)

    # Create empty df with feature names as index
    feat_all_data = []
    empty_df = pd.DataFrame(index=unique_feat_names)
    feat_all_data.append(empty_df)

    # Collect feature weights for all iterations
    for itera in outcomes:
        feature_names = itera[key_feat_names]
        feature_weights = itera[key_feat_weights]
        features_coef_df = pd.DataFrame(feature_weights, index=feature_names)
        feat_all_data.append(features_coef_df)
    # Concatenate the collected DataFrames into a single DataFrame
    feat_all_df = pd.concat(feat_all_data, axis=1, keys=[
        f'itera_{i-1}' for i in range(1, len(outcomes)+2)])

    # Calculate mean feature weights and selection frequencies across iterations
    mean = feat_all_df.mean(axis=1)
    count_na = feat_all_df.isna().sum(axis=1)
    sel_freq = feat_all_df.shape[1] - count_na

    # Save mean feature weights and selection frequenies in DataFrame
    feat_sum_df = pd.DataFrame({
        "selection frequency": sel_freq,
        "mean coefficient": mean},
        index=feat_all_df.index)

    # Sort DataFrame
    feat_sum_df.sort_values(
        by=["selection frequency", "mean coefficient"], key=abs, ascending=False, inplace=True)

    return feat_sum_df

def collect_shaps(outcomes, key_feat_names, key_feat_shaps, key_test_features):
    """
    Collect selected features and shap values across iterations.

    Parameters:
    - outcomes (list): List with one entry per iteration of k-fold cross-validation. 
      Each entry is a dictionary with all information saved per iteration. (results_single_iter)
      This dictionary needs to contain selected features per iteration and their feature weight.
    - key_feat_names (str): Key in the results_single_iter dictionary containing the names of selected features.
    - key_feat_shaps (str): Key in the results_single_iter dictionary containing the shap values of selected features.
    - key_test_features (str): Key in the results_single_iter dictionary containing the feature values of selected features.

    Returns:
    - feat_shaps_all_df: DataFrame collecting, for each feature, the selection frequency and shap values across iterations.
    - feat_values_all_df: DataFrame collecting, for each feature, the values across iterations.
    """
    # Create np_array of features that were at least once part of the final model
    feat_names_all = []
    for itera in outcomes:
        feat_names = itera[key_feat_names]
        feat_names_all.append(feat_names)
    feat_names_all = np.concatenate(feat_names_all)
    
    # Create empty df with feature names as index
    feat_shaps_all_data = []
    feat_values_all_data = []

    # Collect feature weights for all iterations
    for itera in outcomes:
        feature_names = itera[key_feat_names]
        feature_shaps = itera[key_feat_shaps]
        features_shaps_df = pd.DataFrame(feature_shaps, columns=feature_names)
        feat_shaps_all_data.append(features_shaps_df)
    # Concatenate the collected DataFrames into a single DataFrame
    feat_shaps_all_df = pd.concat(feat_shaps_all_data, ignore_index=True)
    
    # Collect feature values for all iterations
    for itera in outcomes:
        feature_names = itera[key_feat_names]
        feature_values = itera[key_test_features]
        features_values_df = pd.DataFrame(feature_values, columns=feature_names)
        feat_values_all_data.append(features_values_df)
    # Concatenate the collected DataFrames into a single DataFrame
    feat_values_all_df = pd.concat(feat_values_all_data, ignore_index=True)

    return feat_shaps_all_df, feat_values_all_df

def make_shap_plots(feat_collect_shap_df, feat_test_values_df, plot_path, treatment_option):
    """
    Creates shap summary bar plot and beeswarm plot for a given treatment option.

    Parameters:
    - feat_sum_shap_df (dataframe): DataFrame collecting, for each feature, the selection frequency and shap values across iterations.
    - feat_test_values_df (dataframe): DataFrame collecting, for each feature, the values across iterations.
    - treatment_option (int): Number of treatement option (0: treatment option A, 1: treatment option B).
    """
    
    #Create necessary arrays
    feature_names = feat_collect_shap_df.columns
    feat_collect_shap_df = np.array(feat_collect_shap_df.fillna(0)) # NaNs replaced by zero for zero value to explanation
    feat_test_values_df = np.array(feat_test_values_df.fillna(0)) # NaNs replaced by zero for zero value to explanation
    
    plt.close('all')
    
    # Create beeswarm summary plot for each treatment option separately
    fig1=plt.gcf()
    if treatment_option == 0:
        shap.summary_plot(feat_collect_shap_df, features = feat_test_values_df, feature_names = feature_names, max_display = 10, plot_type = 'violin', plot_size = (10,10),show=False, layered_violin_max_num_bins=35, title = 'Alternative 0')
        fig1.savefig(os.path.join(plot_path, ("Shap_Beeswarm_treatment_A.png")))
    elif treatment_option == 1:
        shap.summary_plot(feat_collect_shap_df, features = feat_test_values_df, feature_names = feature_names, max_display = 10, plot_type = 'violin', plot_size = (10,10),show=False, layered_violin_max_num_bins=35, title = 'Alternative 1')
        fig1.savefig(os.path.join(plot_path, ("Shap_Beeswarm_treatment_B.png")))
    plt.close('all')

    # Create bar plot for each treatment option separately
    fig2=plt.gcf()
    if treatment_option == 0:
        shap_values_abs = np.abs(feat_collect_shap_df)
        shap.bar_plot(shap_values_abs.mean(0),feature_names = feature_names, max_display = 10,show=False)
        fig2.savefig(os.path.join(plot_path, ("SHAP_Barplot_treatment_A.png")))
    elif treatment_option == 1:
        shap_values_abs = np.abs(feat_collect_shap_df)
        shap.bar_plot(shap_values_abs.mean(0),feature_names = feature_names, max_display = 10,show=False)
        fig2.savefig(os.path.join(plot_path, ("SHAP_Barplot_treatment_B.png")))
