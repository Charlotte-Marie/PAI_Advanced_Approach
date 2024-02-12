# -*- coding: utf-8 -*-
"""
The script defines customized functions and classes for model and PAI evaluation.

@author: charl
"""
# %% Import packages
import numpy as np
import math
from scipy import stats
import os
import seaborn as sns
import matplotlib.pyplot as plt

# %% Evaluate model performance


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

# %% Evaluate PAI


def ev_PAI(y_prediction, plot_path=None, suffix=""):
    """ Evaluate the Personalized Advanatage Index (PAI) and generate visualizations.

    Parameters:
    - y_prediction : pd.DataFrame
      A DataFrame that includes 'y_true' and 'PAI' columns.
    - plot_path : str, optional, default: None
      The directory path to save generated plots. If None, plots will be saved in the current working directory.
    - suffix : str, optional, default: ""
      An additional suffix to append to the plot filenames for better identification.

    Returns:
    - PAI_metrics : dict
      A dictionary containing the calculated PAI metrics and statistical test results.
      - 'mean abspai' : float
        Mean of absolute PAI values.
      - 'mean outcome optimal' : float
        Mean of true outcomes for subjects receiving optimal treatment.
      - 'mean outcome nonoptimal' : float
        Mean of true outcomes for subjects receiving nonoptimal treatment.
      - 't-test_statistic' : float
        T-statistic from the independent two-sample t-test.
      - 't-test_p_value' : float
        P-value from the independent two-sample t-test.
      - 'cohens d' : float
        Cohen's d effect size for the difference in means.
      - 'levene_W_statistic' : float
        W-statistic from the Levene test for variance homogeneity.
      - 'levene_p_value' : float
        P-value from the Levene test for variance homogeneity.
      - 'shapiro_opt_W_statistic' : float
        W-statistic from the Shapiro-Wilk test for normality of outcomes in the optimal group.
      - 'shapiro_opt_p_value' : float
        P-value from the Shapiro-Wilk test for normality of outcomes in the optimal group.
      - 'shapiro_nonopt_W_statistic' : float
        W-statistic from the Shapiro-Wilk test for normality of outcomes in the nonoptimal group.
      - 'shapiro_nonopt_p_value' : float
        P-value from the Shapiro-Wilk test for normality of outcomes in the nonoptimal group.
    """
    if plot_path is None:
        plot_path = os.getcwd()  # Save in the current working directory if plot_path is None

    # Absolute PAIs
    abspai = abs(y_prediction["PAI"])
    abspai_mean = np.mean(abspai)

    # Plot distribution of absolute PAIs
    sns.histplot(abspai)
    plt.xlabel("Absolute PAI")
    plt.savefig(os.path.join(
        plot_path, ("PAI_distribution_" + suffix + ".png")))
    plt.close()

    # Extract values of patients receiving their optimal vs. nonoptimal treatment
    y_prediction['received_treat'] = np.where(
        y_prediction['PAI'] > 0, 'nonoptimal', 'optimal')
    obs_outcomes_optimal = []
    obs_outcomes_nonoptimal = []
    obs_outcomes_optimal = y_prediction[y_prediction['received_treat']
                                        == "optimal"]['y_true'].tolist()
    obs_outcomes_nonoptimal = y_prediction[y_prediction['received_treat']
                                           == "nonoptimal"]['y_true'].tolist()

    # Plot nonoptimal vs. optimal
    sns.histplot(data=y_prediction, x="y_true",
                 hue='received_treat', multiple="dodge")
    plt.savefig(os.path.join(
        plot_path, ("optimal_vs_nonoptimal_" + suffix + ".png")))
    plt.close()

    # Is the difference in means between optimal and nonoptimal significant ?
    # Step 1: Check assumptions
    # 1. Variance homogeneity (significant Levene test -> no variance homogeneity)
    levene_W_statistic, levene_p_value = stats.levene(
        obs_outcomes_nonoptimal, obs_outcomes_optimal, center="mean")
    # 2. Normality (significant Shapiro-Wilk-test -> not normally distributed)
    shapiro_nonopt_W_statistic, shapiro_nonopt_p_value = stats.shapiro(
        obs_outcomes_nonoptimal)
    shapiro_opt_W_statistic, shapiro_opt_p_value = stats.shapiro(
        obs_outcomes_optimal)

    # Mann-Whitney-U-test
    #manwhit_statistic, manwhit_p_value = stats.mannwhitneyu(obs_outcomes_nonoptimal, obs_outcomes_optimal)

    # Step 2: Calculate t-test (even though the normality assumption is often violated,
    # the t-test is used as default as it is robust to most violations)
    mean_outcome_optimal = np.mean(obs_outcomes_optimal)
    mean_outcome_nonoptimal = np.mean(obs_outcomes_nonoptimal)
    t_statistic, t_p_value = stats.ttest_ind(
        obs_outcomes_nonoptimal, obs_outcomes_optimal)

    # Cohen´s d for observed outcome optimal / nonoptimal
    def cohens_d(x, y):
        " Calculate Cohen´s d for two independent groups with unequal sample size"
        n_x = len(x)
        n_y = len(y)
        SD_x = np.std(x)
        SD_y = np.std(y)
        pooled_sd = math.sqrt(
            ((n_x - 1) * SD_x**2 + (n_y - 1) * SD_y**2) / (n_x + n_y - 2))
        d = (np.mean(x) - np.mean(y)) / pooled_sd
        return d

    cohens_d = cohens_d(x=obs_outcomes_nonoptimal, y=obs_outcomes_optimal)

    PAI_metrics = {"mean abspai": abspai_mean,
                   "mean outcome optimal": mean_outcome_optimal,
                   "mean outcome nonoptimal": mean_outcome_nonoptimal,
                   "t-test_statistic": t_statistic,
                   "t-test_p_value": t_p_value,
                   "cohens d": cohens_d,
                   # "manwhit_statistic": manwhit_statistic,
                   # "manwhit_p_value": manwhit_p_value,
                   "levene_W_statistic": levene_W_statistic,
                   "levene_p_value": levene_p_value,
                   "shapiro_opt_W_statistic": shapiro_opt_W_statistic,
                   "shapiro_opt_p_value": shapiro_opt_p_value,
                   "shapiro_nonopt_W_statistic": shapiro_nonopt_W_statistic,
                   "shapiro_nonopt_p_value": shapiro_nonopt_p_value,
                   }

    return PAI_metrics
