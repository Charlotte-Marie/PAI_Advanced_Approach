# -*- coding: utf-8 -*-
"""
The script defines customized functions and classes for PAI evaluation.

@author: charl
"""
# %% Import packages
import numpy as np
import math
from scipy import stats
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# %% Evaluate PAI


def ev_PAI(y_prediction, plot_path=None, suffix="", iteration_num=None):
    """ Evaluate the Personalized Advanatage Index (PAI) and generate visualizations.

    Parameters:
    - y_prediction : pd.DataFrame
      A DataFrame that includes 'y_true' and 'PAI' columns.
    - plot_path : str, optional, default: None
      The directory path to save generated plots. If None, plots will be saved in the current working directory.
    - suffix : str, optional, default: ""
      An additional suffix to append to the plot filenames for better identification.
    - iteration_num : str, optional, default: None
      Current number of iteration of the crossvalidation (to include in plot titles)

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
    abspai_mean = round(np.mean(abspai), 2)

    # Plot distribution of absolute PAIs
    sns.histplot(abspai)
    plt.xlabel("Absolute PAI")
    # Set title
    if iteration_num is not None:
        plt.title(f"Distribution of the absolute PAI (rep. {iteration_num})")
    #Ensure whole numbers
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Save plot
    plt.savefig(os.path.join(
        plot_path, ("PAI_distribution_" + suffix + ".png")))
    plt.close()

    # Extract values of patients receiving their optimal vs. nonoptimal treatment
    y_prediction.loc[:, 'received_treat'] = np.where(
        y_prediction['PAI'] > 0, 'nonoptimal', 'optimal')
    obs_outcomes_optimal = y_prediction[y_prediction['received_treat']
                                        == "optimal"]['y_true'].tolist()
    obs_outcomes_nonoptimal = y_prediction[y_prediction['received_treat']
                                           == "nonoptimal"]['y_true'].tolist()

    # Plot nonoptimal vs. optimal
    sns.histplot(data=y_prediction, x="y_true",
                 hue='received_treat', multiple="dodge",
                 palette={"optimal": "green", "nonoptimal": "red"})
    # Add mean of each group
    mean_optimal = y_prediction[y_prediction['received_treat']
                                == 'optimal']['y_true'].mean()
    mean_nonoptimal = y_prediction[y_prediction['received_treat']
                                   == 'nonoptimal']['y_true'].mean()
    plt.axvline(x=mean_optimal, color='green', linestyle='--')
    plt.axvline(x=mean_nonoptimal, color='red', linestyle='--')
    # Set title
    if iteration_num is not None:
        plt.title(
            f"Distribution of outcome optimal and nonoptimal (rep. {iteration_num})")
    #Ensure whole numbers
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Save plot
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
    # manwhit_statistic, manwhit_p_value = stats.mannwhitneyu(obs_outcomes_nonoptimal, obs_outcomes_optimal)

    # Step 2: Calculate one-sided t-test (even though the normality assumption is often violated,
    # the t-test is used as default as it is robust to most violations)
    mean_outcome_optimal = round(np.mean(obs_outcomes_optimal), 2)
    mean_outcome_nonoptimal = round(np.mean(obs_outcomes_nonoptimal), 2)
    SD_outcome_optimal = round(np.std(obs_outcomes_optimal), 2)
    SD_outcome_nonoptimal = round(np.std(obs_outcomes_nonoptimal), 2)
    n_optimal = len(obs_outcomes_optimal)
    n_nonoptimal = len(obs_outcomes_nonoptimal)
    t_statistic, t_p_value = stats.ttest_ind(
        obs_outcomes_nonoptimal, obs_outcomes_optimal, equal_var=True, alternative="greater")

    # Welch-test
    Welch_t_statistic, Welch_p_value = stats.ttest_ind(
        obs_outcomes_nonoptimal, obs_outcomes_optimal, equal_var=False, alternative="greater")

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

    PAI_metrics = {"mean_abspai": abspai_mean,
                   "mean_outcome_optimal": f"{mean_outcome_optimal} ({SD_outcome_optimal})",
                   "mean_outcome_nonoptimal":  f"{mean_outcome_nonoptimal} ({SD_outcome_nonoptimal})",
                   "n_optimal": n_optimal,
                   "n_nonoptimal": n_nonoptimal,
                   "t-test_statistic": round(t_statistic, 2),
                   "t-test_p_value": round(t_p_value, 4),
                   "cohens d": round(cohens_d, 2),
                   # "manwhit_statistic": manwhit_statistic,
                   # "manwhit_p_value": manwhit_p_value,
                   "levene_W_statistic": round(levene_W_statistic, 2),
                   "levene_p_value": round(levene_p_value, 4),
                   "shapiro_opt_W_statistic": round(shapiro_opt_W_statistic, 2),
                   "shapiro_opt_p_value": round(shapiro_opt_p_value, 4),
                   "shapiro_nonopt_W_statistic": round(shapiro_nonopt_W_statistic, 2),
                   "shapiro_nonopt_p_value": round(shapiro_nonopt_p_value, 4),
                   "Welch_t_statistic": round(Welch_t_statistic, 2),
                   "Welch_t_p_value": round(Welch_p_value, 4)
                   }

    return PAI_metrics

# %% Evaluate PAI across repetitions


def calc_PAI_metrics_across_reps(outcomes, key_PAI_df, n_folds, plot_path):
    """
    Calculate PAI metrics across repetitions of k-fold cross-validation.

    The PAI is evaluated for each repetition of the k-fold cross-validation. For instance,
    if we have performed a 100 * 5-fold cross-validation, the PAI is evaluated 100 times,
    calculating e.g., the mean absolute PAI, testing for a difference in post-treatment severity between patients receiving 
    their optimal vs. nonopitmal treatment, or calculating Cohen´s d.

    Parameters:
    - outcomes (list): List with one entry per iteration. Each entry is a dictionary with all information saved per iteration (results_single_iter).
    - key_PAI_df (str): Key in the results_single_iter dictionary containing the PAI results metrics.
    - n_folds (int): Number of folds in the cross-validation.

    Returns:
    dict: A dictionary containing dataframes with PAI evaluation metrics for each repetition of cross-validation.
        - 'all': Metrics for all PAIs.
        - '50_perc': Metrics for the top 50 percent highest PAIs.
        - 'treat_A': Metrics for PAIs in treatment_A.
        - 'treat_B': Metrics for PAIs in treatment_B.
    Calculate PAI metrics across repetitions of k-fold cross-validation.
    """

    results_PAI_all = []
    results_PAI_50_perc = []
    results_PAI_treat_A = []
    results_PAI_treat_B = []

    # Collect outcomes per repetition
    outcomes_all_repeats = [outcomes[i:i+n_folds]
                            for i in range(0, len(outcomes), n_folds)]

    for repeat, outcomes_one_repeat in enumerate(outcomes_all_repeats):
        y_true_PAI_all_folds = pd.concat(
            [inner_dict[key_PAI_df] for inner_dict in outcomes_one_repeat], ignore_index=True)

        # Calculate Cohens d and absolute PAI for all PAIs
        results_PAI_all.append(ev_PAI(
            y_true_PAI_all_folds,
            plot_path=plot_path,
            suffix=f"{repeat}_all",
            iteration_num=repeat))

        # Calculate Cohens d for 50 percent highest PAIs
        median = np.median(abs(y_true_PAI_all_folds["PAI"]))
        is_50_percent = abs(y_true_PAI_all_folds["PAI"]) > median
        results_PAI_50_perc.append(ev_PAI(y_true_PAI_all_folds[is_50_percent],
                                          plot_path=plot_path,
                                          suffix=f"{repeat}_50_perc",
                                          iteration_num=repeat))

        # Calculate Cohen´s d for single treatments
        results_PAI_treat_A.append(ev_PAI(y_true_PAI_all_folds[y_true_PAI_all_folds["group"] == "treatment_A"],
                                          plot_path=plot_path,
                                          suffix=f"{repeat}_treat_A",
                                          iteration_num=repeat))

        results_PAI_treat_B.append(ev_PAI(y_true_PAI_all_folds[y_true_PAI_all_folds["group"] == "treatment_B"],
                                          plot_path=plot_path,
                                          suffix=f"{repeat}_treat_B",
                                          iteration_num=repeat))

        results_PAI_all_df = pd.DataFrame(results_PAI_all)
        results_PAI_50_perc_df = pd.DataFrame(results_PAI_50_perc)
        results_PAI_treat_A_df = pd.DataFrame(results_PAI_treat_A)
        results_PAI_treat_B_df = pd.DataFrame(results_PAI_treat_B)
        # Add column with index of repetition
        results_PAI_all_df.insert(0, 'repeat', results_PAI_all_df.index)
        results_PAI_50_perc_df.insert(
            0, 'repeat', results_PAI_50_perc_df.index)
        results_PAI_treat_A_df.insert(
            0, 'repeat', results_PAI_treat_A_df.index)
        results_PAI_treat_B_df.insert(
            0, 'repeat', results_PAI_treat_B_df.index)

        # Create common dictionary
        PAI_metrics_across_reps = {
            "all": results_PAI_all_df,
            "50_perc": results_PAI_50_perc_df,
            "treat_A": results_PAI_treat_A_df,
            "treat_B": results_PAI_treat_B_df
        }

    return PAI_metrics_across_reps

# %% Summarize PAI across repetitions


def summarize_PAI_metrics_across_reps(results_PAI_df_dict):
    """
    Summarize PAI evaluation metrics across repetitions.

    Parameters:
    - results_PAI_df_dict (dict): Dictionary with one dataframe with PAI metrics 
      per subgroup (e.g., all, 50 percent highest PAI)

    Returns:
    dict: Dictionary with one dataframe of summary-values per subgroup.

    """
    PAI_metrics_summarized = {}
    for subgroup in results_PAI_df_dict:
        subgroup_df = results_PAI_df_dict[subgroup]
        sum_dict = {}
        sum_dict["n_t_test_sig"] = len(
            subgroup_df[subgroup_df['t-test_p_value'] < 0.05])
        sum_dict["mean abspai"] = np.mean(
            subgroup_df["mean_abspai"])
        sum_dict["mean abspai SD"] = np.std(
            subgroup_df["mean_abspai"])
        sum_dict["mean Cohens d"] = np.mean(
            subgroup_df["cohens d"])
        sum_dict["mean Cohens d SD"] = np.std(
            subgroup_df["cohens d"])
        sum_dict["n_variance_viol"] = len(
            subgroup_df[subgroup_df['levene_p_value'] < 0.05])
        sum_dict["n_normal_viol"] = len(subgroup_df[(subgroup_df['shapiro_opt_p_value'] < 0.05) | (
            subgroup_df['shapiro_nonopt_p_value'] < 0.05)])
        sum_df = pd.DataFrame([sum_dict])  # Turn into dataframe
        PAI_metrics_summarized[subgroup] = sum_df

    return PAI_metrics_summarized
