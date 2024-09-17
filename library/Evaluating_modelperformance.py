# -*- coding: utf-8 -*-
"""
The script defines customized functions and classes for modelperformance evaluation.

@author: Charlotte Meinke, Silvan Hornstein, Rebecca Delfendahl, Till Adam & Kevin Hilbert
"""

# %% Import packages
import numpy as np
import math
import pandas as pd

# %% Evaluate model performance per fold


def calc_modelperformance_metrics(y_prediction):
    """
    Calculate performance metrics based on predicted and true values.

    Parameters:
    - y_prediction : pd.DataFrame
      A DataFrame that includes 'y_true' and 'y_pred_factual' columns.

    Returns:
    - performance_metrics : dict
      A dictionary containing the calculated performance metrics.
      - 'correlation' : float
        Pearson correlation coefficient between 'y_true' and 'y_pred_factual'.
      - 'RMSE' : float
        Root Mean Squared Error between 'y_true' and 'y_pred_factual'.
      - 'MAE' : float
        Mean Absolute Error between 'y_true' and 'y_pred_factual'
    """

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    performance_metrics = {}

    # Correlation
    correlation = np.corrcoef(
        y_prediction['y_pred_factual'], y_prediction['y_true'])[0, 1]
    # Error
    mae = mean_absolute_error(
        y_prediction['y_true'], y_prediction['y_pred_factual'])
    rmse = math.sqrt(mean_squared_error(
        y_prediction['y_true'], y_prediction['y_pred_factual']))

    performance_metrics = {
        "correlation": correlation, "RMSE": rmse, "MAE": mae}

    return performance_metrics

# %% Evaluate modelperformance across folds


def get_modelperformance_metrics_across_folds(outcomes, key_modelperformance_metrics):
    """
    Returns a dataframe with model performance metrics across folds.

    Parameters:
    - outcomes (list): List with one entry per iteration of k-fold cross-validation. 
      Each entry is a dictionary with all information saved per iteration (results_single_iter).
    - key_modelperformance_metrics (str): Name of the dictionary key in the dictionary with multiple results per iteration
    containing the model performance metrics.

    Returns:
    pd.DataFrame: Dataframe with model performance metrics across folds.
    """
    # Turn all modelperformance_metrics dictionaries into dataframes and concatenate them
    dataframes = [pd.DataFrame([inner_dict[key_modelperformance_metrics]])
                  for inner_dict in outcomes]
    modelperformance_metrics_across_folds_df = pd.concat(
        dataframes, ignore_index=True)

    return modelperformance_metrics_across_folds_df


# %% Summarize modelperformance across folds
def summarize_modelperformance_metrics_across_folds(outcomes, key_modelperformance_metrics):
    """
    Summarize model performance metrics across iterations.

    Parameters:
    - outcomes (list): List with one entry per iteration of k-fold cross-validation. 
      Each entry is a dictionary with all information saved per iteration (results_single_iter).
    - key_modelperformance_metrics (str): Key in the results_single_iter dictionary containing the model performance metrics.

    Returns:
    pd.DataFrame: Dataframe with summary statistics for model performance metrics.

    """
    sum_stat_modelperformance_metrics = pd.DataFrame()
    count_iters = len(outcomes)

    for var in outcomes[0][key_modelperformance_metrics]:
        # Concatenate values of all iterations to one list
        list_var = [itera[key_modelperformance_metrics][var]
                    for itera in outcomes]
        # Calculate summary statistics
        if count_iters > 1:
            min_val = min(list_var)
            max_val = max(list_var)
            mean_val = np.mean(list_var)
            std_val = np.std(list_var)
        elif count_iters == 1:
            min_val = "NA"
            max_val = "NA"
            mean_val = list_var[0]
            std_val = "NA"
        # Add summary statistics to the intialized DataFrame
        sum_stat_modelperformance_metrics["Min_" + var] = [min_val]
        sum_stat_modelperformance_metrics["Max_" + var] = [max_val]
        sum_stat_modelperformance_metrics["Mean_" + var] = [mean_val]
        sum_stat_modelperformance_metrics["Std_" + var] = [std_val]

    return sum_stat_modelperformance_metrics
