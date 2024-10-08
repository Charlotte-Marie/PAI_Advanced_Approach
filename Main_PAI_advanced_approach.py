# -*- coding: utf-8 -*-
"""
Created in February 2024
by authors Charlotte Meinke, Silvan Hornstein, Rebecca Delfendahl, Till Adam & Kevin Hilbert
"""

# %% Import packages
# Standard packages
import argparse
import numpy as np
import os
import pandas as pd
import pickle
import shap
import sklearn
import sys
import time
import warnings
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pandas import read_csv
from scipy.sparse import SparseEfficiencyWarning
from scipy.stats import uniform
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Ridge, ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
import xgboost as xgb

# Custom packages 
from library.Evaluating_PAI import calc_PAI_metrics_across_reps, summarize_PAI_metrics_across_reps
from library.Evaluating_feat_importance import summarize_features, collect_shaps, make_shap_plots
from library.Evaluating_modelperformance import calc_modelperformance_metrics, get_modelperformance_metrics_across_folds, summarize_modelperformance_metrics_across_folds
from library.html_script import PAI_to_HTML
from library.Imputing import MiceModeImputer_pipe
from library.Organizing import create_folder_to_save_results, get_categorical_variables
from library.Preprocessing import FeatureSelector
from library.Scaling import ZScalerDimVars


# %% General settings

def set_options_and_paths():
    """ Set options and paths based on command-line or inline arguments depending on the use of command line or the IDE.

    Returns:
    - args: An object containing parsed command-line arguments.
    - PATH_RESULTS: Path to save results.
    """

    def generate_and_create_results_path(args):
        model_name = f"{args.NAME_RESULTS_FOLDER}_{args.CLASSIFIER}"
        if args.HP_TUNING == "True":
            model_name += "_hp_tuned_grid"
        PATH_RESULTS = os.path.join(args.PATH_RESULTS_BASE, model_name)
        create_folder_to_save_results(PATH_RESULTS)
        PATH_RESULTS_PLOTS = os.path.join(PATH_RESULTS, "plots")
        create_folder_to_save_results(PATH_RESULTS_PLOTS)
        PATHS = {
            "RESULT": PATH_RESULTS,
            "RESULT_PLOTS": PATH_RESULTS_PLOTS
        }

        return PATHS

    # Argparser
    parser = argparse.ArgumentParser(
        description='Advanced script to calculate the PAI')
    parser.add_argument('--PATH_INPUT_DATA', type=str,
                        help='Path to input data')
    parser.add_argument('--NAME_RESULTS_FOLDER', type=str,
                        help='Name result folder')
    parser.add_argument('--PATH_RESULTS_BASE', type=str,
                        help='Path to save results')
    parser.add_argument('--NUMBER_FOLDS', type=int, default=5,
                        help='Number of folds in the cross-validation')
    parser.add_argument('--NUMBER_REPETITIONS', type=int, default=100,
                        help='Number of repetitions of the cross-validation')
    parser.add_argument('--CLASSIFIER', type=str,
                        help='Classifier to use, set ridge_regression, random_forest or xgboost')
    parser.add_argument('--HP_TUNING', type=str, default="False",
                        help='Should hyperparameter tuning be applied? Set False or True')
    parser.add_argument('--CALC_SHAP_VALUES', type=str, default="False",
                        help='Should shap values be calculculated? Set False or True')

    args = parser.parse_args()

    try:
        PATHS = generate_and_create_results_path(args)
        print("Using arguments given via terminal or GUI")
    except:
        print("Using arguments given in the script")
        working_directory = os.getcwd()
        args = parser.parse_args([
            '--PATH_INPUT_DATA', os.path.join(
                working_directory, "synthet_test_data"),
            '--NAME_RESULTS_FOLDER', "test_run",
            '--PATH_RESULTS_BASE', working_directory,
            '--CLASSIFIER', 'random_forest',
            '--NUMBER_FOLDS', '5',
            '--NUMBER_REPETITIONS', '2',
            '--CALC_SHAP_VALUES', 'True',
        ])
        PATHS = generate_and_create_results_path(args)

    return args, PATHS


def generate_treatstratified_splits(PATH_INPUT_DATA, n_folds, n_repeats):
    """Generate splits startitied for treatment groups.

    Args:
    PATH_INPUT_DATA (str): Path to the input data directory.
    n_folds (int): Number of folds in the cross-validation.
    n_repeats (int): Number of repetitions of the cross-validation.

    Returns:
    splits (list of tuples): List containing train and test indices.
    """
    groups_import_path = os.path.join(PATH_INPUT_DATA, "groups.txt")
    groups = read_csv(groups_import_path, sep="\t", header=0)
    y = np.array(groups)
    sfk = RepeatedStratifiedKFold(n_splits=n_folds,
                                  n_repeats=n_repeats,
                                  random_state=0)
    splits = list(sfk.split(np.zeros(len(y)), y))
    return splits

# %% Handle Warnings 
# FutureWarning cannot be addressed directly since it is a library-level warning. Make sure seaborn and pandas are up-to-date! 
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# SparseEfficiencyWarning
# Also seems to be on library-level
# related to how scikit-learn or other libraries handle sparse matrices internally
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)

# RuntimeWarning
# Warning = np.nanmean is called on a slice of an array that contains only NaN values
# RuntimeWarning just informs that whole slices of the array contain NaNs and skips over them with no change to the subsequent analysis
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %% Procedure for one iteration/split in the repeated stratified cross-validation

def procedure_per_iter(split, PATH_RESULTS, PATH_INPUT_DATA, args):
    """
    Perform a single iteration in the repeated k-fold cross-valdiation

    Parameters:
    - split: Tuple containing train and test indices.
    - PATH_RESULTS: Path to save results.
    - PATH_INPUT_DATA: Path to the input data folder.
    - args: Argument object containing arguments for running the script

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
    
    # Load categorical variables names if they are available
    # Otherwise the Imputer does automatically identify string variables as categorical variables
    try:
        catvars_import_path = os.path.join(PATH_INPUT_DATA, "categorical_vars.txt")
        names_categorical_vars = get_categorical_variables(catvars_import_path)
    except: 
        names_categorical_vars = None

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
    X_train_all_treat_imp = imputer.fit_transform(X_train_all_treat, names_categorical_vars)
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
        if args.HP_TUNING == "True":
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
        if args.CLASSIFIER == "random_forest":
            if args.HP_TUNING == "True":
                parameters = {"max_depth": [3, 4, 5],
                              "max_features": ['sqrt', 'log2'], 
                              "min_samples_split": [2, 6, 10],
                              "min_samples_leaf": [1, 3, 5]}
                clf_hp = RandomForestRegressor(n_estimators=100, criterion='squared_error', bootstrap=True, oob_score=False, random_state=random_state_seed)
                grid_rf = GridSearchCV(
                    clf_hp, parameters, scoring='neg_mean_absolute_error', cv=5)
                grid_rf.fit(X_train_scaled, y_train)
                background_hp_tuning_rf = pd.DataFrame(
                    grid_rf.cv_results_)
                # Add to info_per_treament
                info_per_treat[treatment]["background_rf_hp"] = background_hp_tuning_rf
                clf = grid_rf.best_estimator_
            else:
                clf = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                            max_depth=None, min_samples_split=5,
                                            max_features=1.0, bootstrap=True,
                                            oob_score=False, random_state=random_state_seed,
                                            max_samples=None)
            clf.fit(X_train_scaled_selected_factual, y_train)
            feature_weights = clf.feature_importances_
            # Get SHAP values for treatment
            if args.CALC_SHAP_VALUES == "True":
                shap_explainer = shap.TreeExplainer(model = clf, data = X_train_scaled_selected_factual, model_output='raw', feature_perturbation='interventional')
                shap_values = shap_explainer.shap_values(X_test_scaled_selected_factual, check_additivity=False)
            else:
                shap_values = None
            
        elif args.CLASSIFIER == "ridge_regression":
            if args.HP_TUNING == "True":
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
            # Get SHAP values for treatment
            if args.CALC_SHAP_VALUES == "True":
                masker = shap.maskers.Independent(data = X_train_scaled_selected_factual)
                shap_explainer = shap.LinearExplainer(model = clf, masker = masker, data = X_train_scaled_selected_factual)
                shap_values = shap_explainer.shap_values(X_test_scaled_selected_factual) 
            else:
                shap_values = None
        elif args.CLASSIFIER == "xgboost":
            if args.HP_TUNING == "True":
                parameters = {"learning_rate": uniform(0.03, 0.3), 
                               "max_depth": [2, 5, 10],
                               "colsample_bytree": uniform(0.7, 0.3), 
                               "gamma": uniform(0, 0.5),
                               "reg_alpha": [0, 0.01, 0.1, 1, 10],  
                               "reg_lambda": [0.01, 0.1, 1, 10], 
                               "subsample": uniform(0.6, 0.4)}
                clf_hp = xgb.XGBRegressor(booster='gbtree', n_estimators=100, objective="reg:squarederror", random_state=random_state_seed)
                grid_xgb = GridSearchCV(
                    clf_hp, parameters, scoring='neg_mean_absolute_error', cv=5)
                grid_xgb.fit(X_train_scaled, y_train)
                background_hp_tuning_xgb = pd.DataFrame(
                    grid_xgb.cv_results_)
                # Add to info_per_treament
                info_per_treat[treatment]["background_xgb_hp"] = background_hp_tuning_xgb
                clf = grid_xgb.best_estimator_
            else:
                clf = xgb.XGBRegressor(booster='gbtree', n_estimators=100, learning_rate=0.1, max_depth=5, gamma=0, colsample_bytree=0.5, objective="reg:squarederror", reg_alpha=0, reg_lambda=1, subsample=0.5, random_state=random_state_seed)
            clf.fit(X_train_scaled_selected_factual, y_train)
            feature_weights = clf.feature_importances_
            # Get SHAP values for treatment
            if args.CALC_SHAP_VALUES == "True":
                shap_explainer = shap.TreeExplainer(model = clf, data = X_train_scaled_selected_factual, model_output='raw', feature_perturbation='interventional')
                shap_values = shap_explainer.shap_values(X_test_scaled_selected_factual) 
            else: 
                shap_values = None

        # Make predictions on the test-set for the factual treatment and save more information for later
        y_true_pred_df_one_fold = pd.DataFrame()
        y_true_pred_df_one_fold["y_pred_factual"] = clf.predict(
            X_test_scaled_selected_factual)
        y_true_pred_df_one_fold["y_true"] = y_test[:]
        y_true_pred_df_one_fold["group"] = treatment

        # Save information per treatment
        new_data = {
            "X_test": X_test_cleaned,
            "X_test_selected": X_test_scaled_selected_factual,
            "fitted scaler": scaler,
            "fitted selector": sfm,
            "clf": clf,
            "n_feat_in_clf": clf.n_features_in_,
            "feature_names": feature_names_selected,
            "feature_weights": feature_weights,
            "shap_values": shap_values,
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

    # Calculate model performance metrics seperately and across metrics
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
        "sel_features_shap_treat_A": info_per_treat["treatment_A"]["shap_values"],
        "sel_features_names_treat_B": info_per_treat["treatment_B"]["feature_names"],
        "sel_features_coef_treat_B": info_per_treat["treatment_B"]["feature_weights"],
        "sel_features_shap_treat_B": info_per_treat["treatment_B"]["shap_values"],
        "n_feat_treat_A": info_per_treat["treatment_A"]["n_feat_in_clf"],
        "n_feat_treat_B": info_per_treat["treatment_B"]["n_feat_in_clf"],
        "excluded_feat": feat_names_excluded,
        "all_features": feat_names_X_imp,
        "test_feature_values_treat_A": info_per_treat["treatment_A"]["X_test_selected"], 
        "test_feature_values_treat_B": info_per_treat["treatment_B"]["X_test_selected"]
    }
    # Add other variables in case of hp tuning
    if args.HP_TUNING == "True":
        if args.CLASSIFIER == "random_forest":
            info_hp = {
                "en_background_treat_A": info_per_treat["treatment_A"]["background_en_hp"],
                "en_background_treat_B": info_per_treat["treatment_B"]["background_en_hp"],
                "rf_background_treat_A": info_per_treat["treatment_A"]["background_rf_hp"],
                "rf_background_treat_B": info_per_treat["treatment_B"]["background_rf_hp"]
            }        
        elif args.CLASSIFIER == "ridge_regression":
            info_hp = {
                "en_background_treat_A": info_per_treat["treatment_A"]["background_en_hp"],
                "en_background_treat_B": info_per_treat["treatment_B"]["background_en_hp"],
                "ridge_background_treat_A": info_per_treat["treatment_A"]["background_ridge_hp"],
                "ridge_background_treat_B": info_per_treat["treatment_B"]["background_ridge_hp"]
            }            
        elif args.CLASSIFIER == "xgboost":
            info_hp = {
                "en_background_treat_A": info_per_treat["treatment_A"]["background_en_hp"],
                "en_background_treat_B": info_per_treat["treatment_B"]["background_en_hp"],
                "xgb_background_treat_A": info_per_treat["treatment_A"]["background_xgb_hp"],
                "xgb_background_treat_B": info_per_treat["treatment_B"]["background_xgb_hp"]
            }
        results_single_iter.update(info_hp)

    return results_single_iter

# %% Run main script
if __name__ == '__main__':

    start_time = time.time()
    print('\nThe scikit-learn version is {}.'.format(sklearn.__version__))

    args, PATHS = set_options_and_paths()

    # Perform splitting stratified by treatment group
    splits = generate_treatstratified_splits(args.PATH_INPUT_DATA,
                                             n_folds=args.NUMBER_FOLDS,
                                             n_repeats=args.NUMBER_REPETITIONS)

    # Run procedure per iterations
    procedure_per_iter_spec = partial(procedure_per_iter,
                                      PATH_RESULTS=PATHS["RESULT"],
                                      PATH_INPUT_DATA=args.PATH_INPUT_DATA,
                                      args=args)
    outcomes = []

    # Multiprocessing (on cluster or local computer)
    pool = Pool(16)
    outcomes[:] = pool.map(procedure_per_iter_spec, splits)
    pool.close()
    pool.join()
    # outcomes[:] = map(procedure_per_iter_spec,splits)  #  no multiprocessing

    # Save outcomes
    with open(os.path.join(PATHS["RESULT"], 'outcomes.pkl'), 'wb') as file:
        pickle.dump(outcomes, file)
    # with open(os.path.join(PATHS["RESULT"], 'outcomes.pkl'), 'rb') as file:
       # outcomes = pickle.load(file)

    # Summarize results across folds or repetitions of k-fold cross-validation
    modelperformance_metrics_across_folds = get_modelperformance_metrics_across_folds(
        outcomes, key_modelperformance_metrics="modelperformance_metrics")
    modelperformance_metrics_summarized = summarize_modelperformance_metrics_across_folds(
        outcomes, key_modelperformance_metrics="modelperformance_metrics")
    PAI_metrics_across_reps = calc_PAI_metrics_across_reps(
        outcomes, key_PAI_df="y_true_PAI", n_folds=args.NUMBER_FOLDS,
        plot_path=PATHS["RESULT_PLOTS"])
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
        PATHS["RESULT"], "modelperformance_across_folds.txt"), sep="\t", na_rep="NA")
    modelperformance_metrics_summarized.T.to_csv(os.path.join(
        PATHS["RESULT"], "modelperformance_summary.txt"), sep="\t")
    for subgroup in PAI_metrics_across_reps:
        PAI_metrics_across_reps[subgroup].to_csv(os.path.join(
            PATHS["RESULT"], ("PAI_across_repetitions_" + subgroup + ".txt")), sep="\t", na_rep="NA")
    for subgroup in PAI_metrics_summarized:
        PAI_metrics_summarized[subgroup].to_csv(os.path.join(
            PATHS["RESULT"], ("PAI_summary_" + subgroup + ".txt")), sep="\t", na_rep="NA")
    feat_sum_treat_A.to_csv(os.path.join(
        PATHS["RESULT"], "features_sum_treat_A.txt"), sep="\t", na_rep="NA")
    feat_sum_treat_B.to_csv(os.path.join(
        PATHS["RESULT"], "features_sum_treat_B.txt"), sep="\t", na_rep="NA")
  
    
    # Summarize and save SHAP values and create plots
    if args.CALC_SHAP_VALUES == "True":
        shap_treat_A, feat_values_treat_A = collect_shaps(outcomes=outcomes,
                                              key_feat_names="sel_features_names_treat_A",
                                              key_feat_shaps="sel_features_shap_treat_A",
                                              key_test_features="test_feature_values_treat_A")
        shap_treat_B, feat_values_treat_B = collect_shaps(outcomes=outcomes,
                                              key_feat_names="sel_features_names_treat_B",
                                              key_feat_shaps="sel_features_shap_treat_B",
                                              key_test_features="test_feature_values_treat_B")
        shap_treat_A.to_csv(os.path.join(
            PATHS["RESULT"], "shap_treat_A.txt"), sep="\t", na_rep="NA")
        shap_treat_B.to_csv(os.path.join(
            PATHS["RESULT"], "shap_treat_B.txt"), sep="\t", na_rep="NA")
        make_shap_plots(shap_treat_A, feat_values_treat_A, plot_path=PATHS["RESULT_PLOTS"], treatment_option = 0)
        make_shap_plots(shap_treat_B, feat_values_treat_B, plot_path=PATHS["RESULT_PLOTS"], treatment_option = 1)       
    
    # HTML Summary
    try:
        PAI_to_HTML(PATHS["RESULT"], plots_directory=PATHS["RESULT_PLOTS"],
                    number_folds=args.NUMBER_FOLDS, number_repetit=args.NUMBER_REPETITIONS)
        print("HTML output successfully created and saved to HTML_output folder")
    except:
        print("Failed to create HTML output")

    elapsed_time = time.time() - start_time
    print('\nThe time for running was {}.'.format(elapsed_time))
    print('Results were saved at {}.'.format(PATHS["RESULT"]))
