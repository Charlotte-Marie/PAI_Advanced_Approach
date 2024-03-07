# -*- coding: utf-8 -*-
"""
Created in February 2024
by authors Charlotte Meinke, Kevin Hilbert & Silvan Hornstein
"""

# %% Import packages
from library.Imputing import MiceModeImputer_pipe
from library.Preprocessing import FeatureSelector, ZScalerDimVars
from library.Evaluating_PAI import calc_PAI_metrics_across_reps, summarize_PAI_metrics_across_reps
from library.Evaluating_modelperformance import calc_modelperformance_metrics, get_modelperformance_metrics_across_folds, summarize_modelperformance_metrics_across_folds
from library.Evaluating_feat_importance import summarize_features
from library.Organizing import create_folder_to_save_results
import pickle
import os
import sys
import time
from multiprocessing import Pool

import sklearn

import numpy as np
from collections import Counter
import pandas as pd
from pandas import read_csv
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Ridge, ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from functools import partial
import argparse

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# %% Functions for generell settings


def set_options(classifier, number_folds=5, number_repetit=100, hp_tuning="False"):
    """ Set options """
    OPTIONS = {}
    OPTIONS['number_folds'] = number_folds
    OPTIONS["number_repetit"] = number_repetit
    OPTIONS["classifier"] = classifier
    OPTIONS["hp_tuning"] = hp_tuning

    return OPTIONS


def generate_and_create_results_path(path_results_base, name_results_folder, OPTIONS):
    # Generate PATH_RESULTS
    if OPTIONS["hp_tuning"] == "True":
        final_name_results_folder = name_results_folder + "_" + \
            OPTIONS["classifier"] + "_" + "hp_tuned_grid"
    else:
        final_name_results_folder = name_results_folder + \
            "_" + OPTIONS["classifier"]
    PATH_RESULTS = os.path.join(path_results_base, final_name_results_folder)
    PATH_RESULTS_PLOTS = os.path.join(PATH_RESULTS, "plots")
    create_folder_to_save_results(PATH_RESULTS)
    create_folder_to_save_results(PATH_RESULTS_PLOTS)

    return PATH_RESULTS

# %% Generate splits of cross-validation


def generate_treatstratified_splits(PATH_INPUT_DATA, n_folds, n_repeats):
    groups_import_path = os.path.join(PATH_INPUT_DATA, "groups.txt")
    groups = read_csv(groups_import_path, sep="\t", header=0)
    y = np.array(groups)
    sfk = RepeatedStratifiedKFold(n_splits=n_folds,
                                  n_repeats=n_repeats,
                                  random_state=0)
    splits = list(sfk.split(np.zeros(len(y)), y))
    return splits


# %% Procedure for one iteration/split in the repeated stratified cross-validation


def procedure_per_iter(split, PATH_RESULTS, PATH_INPUT_DATA, OPTIONS):
    """
    Perform a single iteration in the repeated k-fold cross-valdiation

    Parameters:
    - split: Tuple containing train and test indices.
    - features_path: Path to the features file.
    - labels_path: Path to the labels file.
    - groups_id_path: Path to the groups ID file.
    - options: Dictionary containing various options.

    Returns:
    - results_single_iter: Dictionary containing results for the iteration.
    """

    random_state_seed = 0

    # Load dataset
    features_import_path = os.path.join(PATH_INPUT_DATA, "features.txt")
    labels_import_path = os.path.join(PATH_INPUT_DATA, "labels.txt")
    groups_import_path = os.path.join(PATH_INPUT_DATA, "groups.txt")

    features_import = read_csv(features_import_path, sep="\t", header=0)
    labels_import = read_csv(labels_import_path, sep="\t", header=0)
    # Sanity check
    # features_import["outcome"] = labels_import
    name_groups_id_import = read_csv(
        groups_import_path, sep="\t", header=0)
    groups = np.ravel(np.array(name_groups_id_import))
    y = np.array(labels_import)
    X_df = features_import

    # Perform splitting of dataframe into training and testset
    train_index = split[0]
    test_index = split[1]
    X_train_all_treat, X_test_all_treat = X_df.loc[train_index], X_df.loc[test_index]
    y_train_all_treat, y_test_all_treat = y[train_index], y[test_index]
    groups_train = groups[train_index]
    groups_test = groups[test_index]

    # Deal with missings (Remove variables with too many missings and
    # impute missings in remaining variables)
    imputer = MiceModeImputer_pipe()
    X_train_all_treat_imp = imputer.fit_transform(X_train_all_treat)
    X_test_all_treat_imp = imputer.transform(X_test_all_treat)
    feat_names_X_imp = imputer.new_feat_names

    # Exclude features using FeatureSelector across treatments
    selector = FeatureSelector()
    selector.fit(X_train_all_treat_imp)
    X_train_cleaned_all_treat = selector.transform(X_train_all_treat_imp)
    X_test_cleaned_all_treat = selector.transform(X_test_all_treat_imp)
    feature_names_clean = feat_names_X_imp[selector.is_feat_excluded == 0]
    feat_names_excluded = feat_names_X_imp[selector.is_feat_excluded == 1]

    # For each treatment, train a model on those patients in the training set
    # who have received the treatment ( = factual data)
    # Initialize empty dictionaries to save information per treatment
    # Treatment A is always the more frequent one
    counts_groups = Counter(groups)
    group_labels = sorted(counts_groups, key=counts_groups.get, reverse=True)
    info_per_treat = {"treatment_A": {
        "label": group_labels[0]}, "treatment_B": {"label": group_labels[1]}}

    for treatment in info_per_treat:

        # Subset traing and test set to those patients who received the treatment of interest
        # (= factual data)
        label = info_per_treat[treatment]["label"]
        X_train_cleaned = X_train_cleaned_all_treat[groups_train == label, :]
        X_test_cleaned = X_test_cleaned_all_treat[groups_test == label, :]
        y_train = np.ravel(y_train_all_treat[groups_train == label])
        y_test = np.ravel(y_test_all_treat[groups_test == label])

        # Z-Scale dimensional columns
        scaler = ZScalerDimVars()
        X_train_scaled = scaler.fit_transform(X_train_cleaned)
        X_test_scaled = scaler.transform(X_test_cleaned)

        # Select features with Elastic net
        if OPTIONS["hp_tuning"] == "True":
            parameters = {'alpha': [0.01, 0.1, 1, 10]}
            clf = ElasticNet(l1_ratio=0.5, fit_intercept=False,
                             max_iter=1000, tol=0.0001, random_state=random_state_seed, selection='cyclic')
            grid_en = GridSearchCV(
                clf, parameters, scoring='neg_mean_absolute_error', cv=5)
            grid_en.fit(X_train_scaled, y_train)
            background_hp_tuning = pd.DataFrame(grid_en.cv_results_)
            # Add to info_per_treament
            info_per_treat[treatment]["background_en_hp"] = background_hp_tuning
            # Initiate model with best estimator
            clf_elastic = grid_en.best_estimator_
            # clf_elastic = ElasticNetCV(cv = 5, fit_intercept=False,
            #                                          max_iter=1000, tol=0.0001, random_state=random_state_seed, selection='cyclic')
        else:
            clf_elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=False,
                                     max_iter=1000, tol=0.0001, random_state=random_state_seed, selection='cyclic')
        sfm = SelectFromModel(clf_elastic, threshold="mean")
        sfm.fit(X_train_scaled, y_train)
        is_selected = sfm.get_support()
        feature_names_selected = feature_names_clean[is_selected]
        X_train_scaled_selected_factual = sfm.transform(X_train_scaled)
        X_test_scaled_selected_factual = sfm.transform(X_test_scaled)

        # Fit classifier
        if OPTIONS["classifier"] == "random_forest":
            clf = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                        max_depth=None, min_samples_split=5,
                                        max_features=1.0, bootstrap=True,
                                        oob_score=False, random_state=random_state_seed,
                                        max_samples=None)
            clf.fit(X_train_scaled_selected_factual, y_train)
            feature_weights = clf.feature_importances_
        elif OPTIONS["classifier"] == "ridge_regression":
            if OPTIONS["hp_tuning"] == "True":
                parameters = {'alpha': [0.01, 0.1, 1, 10, 20, 30, 40, 50]}
                clf_hp = Ridge(fit_intercept=False)
                grid_ridge = GridSearchCV(
                    clf_hp, parameters, scoring='neg_mean_absolute_error', cv=5)
                grid_ridge.fit(X_train_scaled, y_train)
                background_hp_tuning_ridge = pd.DataFrame(
                    grid_ridge.cv_results_)
                # Add to info_per_treament
                info_per_treat[treatment]["background_ridge_hp"] = background_hp_tuning_ridge
                clf = grid_ridge.best_estimator_
                # clf = RidgeCV(fit_intercept = False)
            else:
                clf = Ridge(fit_intercept=False)
            clf.fit(X_train_scaled_selected_factual, y_train)
            feature_weights = clf.coef_

        # Make predictions on the test-set for the factual treatment and save more information for later
        y_true_pred_df_one_fold = pd.DataFrame()
        y_true_pred_df_one_fold["y_pred_factual"] = clf.predict(
            X_test_scaled_selected_factual)
        y_true_pred_df_one_fold["y_true"] = y_test[:]
        y_true_pred_df_one_fold["group"] = treatment

        # Save information per treatment
        new_data = {
            "X_test": X_test_cleaned,
            "fitted scaler": scaler,
            "fitted selector": sfm,
            "clf": clf,
            "n_feat_in_clf": clf.n_features_in_,
            "feature_names": feature_names_selected,
            "feature_weights": feature_weights,
            "y_true_pred_df_one_fold": y_true_pred_df_one_fold
        }
        info_per_treat[treatment].update(new_data)

    # Add predictions for the counterfactual treatment
    for treatment in info_per_treat:
        y_true_pred_df_one_fold = info_per_treat[treatment]["y_true_pred_df_one_fold"]
        X_test_counterf = info_per_treat[treatment]["X_test"]

        # Apply imputer, scaler, and feature selector fitted on the other treatment
        # on the counterfactual testset
        other_treatment = next(t for t in info_per_treat if t != treatment)
        scaler_other = info_per_treat[other_treatment]["fitted scaler"]
        selector_other = info_per_treat[other_treatment]["fitted selector"]
        clf_other = info_per_treat[other_treatment]["clf"]
        X_test_counterf_scaled = scaler_other.transform(X_test_counterf)
        X_test_counterf_scaled_selected = selector_other.transform(
            X_test_counterf_scaled)

        # Make predictions for the counterfacutal treatment and calculate the PAI
        y_true_pred_df_one_fold["y_pred_counterfactual"] = clf_other.predict(
            X_test_counterf_scaled_selected)
        y_true_pred_df_one_fold["PAI"] = y_true_pred_df_one_fold['y_pred_factual'] - \
            y_true_pred_df_one_fold['y_pred_counterfactual']

        info_per_treat[treatment]["y_true_pred_df_one_fold"] = y_true_pred_df_one_fold

    # Calculate model performance metrics seperately and across treatments
    modelperformance_metrics_t_A = calc_modelperformance_metrics(
        info_per_treat["treatment_A"]["y_true_pred_df_one_fold"])
    modelperformance_metrics_t_B = calc_modelperformance_metrics(
        info_per_treat["treatment_B"]["y_true_pred_df_one_fold"])
    y_true_pred_df_t_all = pd.concat([info_per_treat["treatment_A"]["y_true_pred_df_one_fold"],
                                      info_per_treat["treatment_B"]["y_true_pred_df_one_fold"]
                                      ], ignore_index=True)
    modelperformance_metrics_all = calc_modelperformance_metrics(
        y_true_pred_df_t_all)
    modelperformance_metrics = {}
    prefixes = ["all", "option_A", "option_B"]
    for prefix, d in zip(prefixes, [modelperformance_metrics_all,
                                    modelperformance_metrics_t_A, modelperformance_metrics_t_B]):
        for key, value in d.items():
            new_key = f'{prefix}_{key}'  # Add the prefix
            modelperformance_metrics[new_key] = value

    # Save relevant information for each iteration in a dictionary
    results_single_iter = {
        "y_true_PAI": y_true_pred_df_t_all[["y_true", "PAI", "group"]],
        "modelperformance_metrics": modelperformance_metrics,
        "sel_features_names_treat_A": info_per_treat["treatment_A"]["feature_names"],
        "sel_features_coef_treat_A": info_per_treat["treatment_A"]["feature_weights"],
        "sel_features_names_treat_B": info_per_treat["treatment_B"]["feature_names"],
        "sel_features_coef_treat_B": info_per_treat["treatment_B"]["feature_weights"],
        "n_feat_treat_A": info_per_treat["treatment_A"]["n_feat_in_clf"],
        "n_feat_treat_B": info_per_treat["treatment_B"]["n_feat_in_clf"],
        "excluded_feat": feat_names_excluded
    }
    # Add other variables in case of hp tuning
    if OPTIONS["hp_tuning"] == "True":
        info_hp = {
            "en_background_treat_A": info_per_treat["treatment_A"]["background_en_hp"],
            "en_background_treat_B": info_per_treat["treatment_B"]["background_en_hp"],
            "ridge_background_treat_A": info_per_treat["treatment_A"]["background_ridge_hp"],
            "ridge_background_treat_B": info_per_treat["treatment_B"]["background_ridge_hp"]
        }
        results_single_iter.update(info_hp)

    return results_single_iter

# %% Function to summarize results across iterations


def aggregate_and_save_results(outcomes):
    # Summarize results across folds or repetitions of k-fold cross-validation
    modelperformance_metrics_across_folds = get_modelperformance_metrics_across_folds(
        outcomes, key_modelperformance_metrics="modelperformance_metrics")
    modelperformance_metrics_summarized = summarize_modelperformance_metrics_across_folds(
        outcomes, key_modelperformance_metrics="modelperformance_metrics")
    PAI_metrics_across_reps = calc_PAI_metrics_across_reps(
        outcomes, key_PAI_df="y_true_PAI", n_folds=OPTIONS["number_folds"],
        plot_path=os.path.join(PATH_RESULTS, "plots"))
    PAI_metrics_summarized = summarize_PAI_metrics_across_reps(
        PAI_metrics_across_reps)
    feat_sum_treat_A = summarize_features(outcomes=outcomes,
                                          key_feat_names="sel_features_names_treat_A",
                                          key_feat_weights="sel_features_coef_treat_A")
    feat_sum_treat_B = summarize_features(outcomes=outcomes,
                                          key_feat_names="sel_features_names_treat_B",
                                          key_feat_weights="sel_features_coef_treat_B")

    # Save summaries as csv
    modelperformance_metrics_across_folds.to_csv(os.path.join(
        PATH_RESULTS, "modelperformance_across_folds.txt"), sep="\t", na_rep="NA")
    modelperformance_metrics_summarized.T.to_csv(os.path.join(
        PATH_RESULTS, "modelperformance_summary.txt"), sep="\t")
    for subgroup in PAI_metrics_across_reps:
        df = PAI_metrics_across_reps[subgroup]
        path = os.path.join(
            PATH_RESULTS, ("PAI_across_repetitions_" + subgroup + ".txt"))
        df.to_csv(path, sep="\t", na_rep="NA")
    for subgroup in PAI_metrics_summarized:
        df = PAI_metrics_summarized[subgroup]
        path = os.path.join(PATH_RESULTS, ("PAI_summary_" + subgroup + ".txt"))
        df.to_csv(path, sep="\t", na_rep="NA")
    feat_sum_treat_A.to_csv(os.path.join(
        PATH_RESULTS, "features_sum_treat_A.txt"), sep="\t", na_rep="NA")
    feat_sum_treat_B.to_csv(os.path.join(
        PATH_RESULTS, "features_sum_treat_B.txt"), sep="\t", na_rep="NA")


# %% Run main script
if __name__ == '__main__':

    # Run script via IDE (start)
    # working_directory = os.getcwd()
    # path_data = os.path.join(working_directory, "synthet_test_data")
    # path_results_base = working_directory
    # PATH_INPUT_DATA = path_data
    # OPTIONS = set_options(classifier = "random_forest",
    #                       number_folds = 5,
    #                       number_repetit = 2,
    #                       hp_tuning = "false"
    #                       )
    # PATH_RESULTS = generate_and_create_results_path(path_results_base,
    #                                                 name_results_folder = "sdfsdf",
    #                                                 OPTIONS = OPTIONS)
    # Run script via IDE (end)

    # Run script via terminal or GUI (start)
    parser = argparse.ArgumentParser(
        description='Advanced script to calculate the PAI')
    parser.add_argument('--PATH_INPUT_DATA', type=str,
                        help='Specify path to folder containing input data')
    parser.add_argument('--NAME_RESULTS_FOLDER', type=str,
                        help='Specify name for results folder')
    parser.add_argument('--PATH_RESULTS_BASE', type=str,
                        help='Specify directory for results folder')
    parser.add_argument('--NUMBER_FOLDS', type=int, default=5,
                        help='Number of folds in the cross-validation')
    parser.add_argument('--NUMBER_REPETIT', type=int, default=1,
                        help='Number of repetitions of the cross-validation')
    parser.add_argument('--CLASSIFIER', type=str,
                        help='Classifier to use, set ridge_regression or random_forest')
    parser.add_argument('--HP_TUNING', type=str, default="False",
                        help='Should hyperparameter tuning be applied? Set False or True')
    args = parser.parse_args()

    PATH_INPUT_DATA = args.PATH_INPUT_DATA
    OPTIONS = set_options(classifier=args.CLASSIFIER,
                          number_folds=args.NUMBER_FOLDS,
                          number_repetit=args.NUMBER_REPETIT,
                          hp_tuning=args.HP_TUNING
                          )
    PATH_RESULTS = generate_and_create_results_path(path_results_base=args.PATH_RESULTS_BASE,
                                                    name_results_folder=args.NAME_RESULTS_FOLDER,
                                                    OPTIONS=OPTIONS)
    # Run script via terminal or GUI (end)

    # Set-up
    start_time = time.time()
    print('\nThe scikit-learn version is {}.'.format(sklearn.__version__))

    # Perform splitting stratified by treatment group
    splits = generate_treatstratified_splits(PATH_INPUT_DATA,
                                             n_folds=OPTIONS['number_folds'],
                                             n_repeats=OPTIONS["number_repetit"])

    # Run procedure per iterations
    # Specify function procedure_per_iter to use it in pooling
    procedure_per_iter_spec = partial(procedure_per_iter,
                                      PATH_RESULTS=PATH_RESULTS,
                                      PATH_INPUT_DATA=PATH_INPUT_DATA,
                                      OPTIONS=OPTIONS)
    outcomes = []
    # Multiprocessing (on cluster or local computer)
    pool = Pool(16)
    outcomes[:] = pool.map(procedure_per_iter_spec, splits)
    pool.close()
    pool.join()
    # outcomes[:] = map(procedure_per_iter_spec,splits)  #local computer

    # Save outcomes
    with open(os.path.join(PATH_RESULTS, 'outcomes.pkl'), 'wb') as file:
        pickle.dump(outcomes, file)
    with open(os.path.join(PATH_RESULTS, 'outcomes.pkl'), 'rb') as file:
        outcomes = pickle.load(file)

    # Aggregate results
    aggregate_and_save_results(outcomes)

    elapsed_time = time.time() - start_time
    print('\nThe time for running was {}.'.format(elapsed_time))
    print('Results were saved at {}.'.format(PATH_RESULTS))
