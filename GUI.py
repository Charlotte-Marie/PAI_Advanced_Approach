# -*- coding: utf-8 -*-
"""
This script create a simple graphical user interface for running the Main script.

@author: Charlotte Meinke, Silvan Hornstein, Rebecca Delfendahl, Till Adam & Kevin Hilbert
"""
import os
import tkinter as tk
from tkinter import Entry, Label, OptionMenu, IntVar, StringVar, Button, filedialog
import subprocess
import sys

# %% Functions


def choose_input_data_path():
    path = filedialog.askdirectory()
    entry_path_input_data.delete(0, tk.END)
    entry_path_input_data.insert(0, path)


def choose_results_base_path():
    path = filedialog.askdirectory()
    entry_path_results_base.delete(0, tk.END)
    entry_path_results_base.insert(0, path)


def run_script():
    print("...is running...")
    current_script_path = os.path.realpath(__file__)
    script_path = os.path.join(os.path.dirname(
        current_script_path), "Main_PAI_advanced_approach.py")

    # Get user-provided arguments from entry widgets
    name_results_folder = entry_name_results_folder.get()
    path_input_data = entry_path_input_data.get()
    path_results_base = entry_path_results_base.get()
    number_folds = number_folds_var.get()
    number_repetit = number_repetit_var.get()
    classifier = classifier_var.get()
    hp_tuning = hp_tuning_var.get()
    SHAP = SHAP_var.get()

    # Construct the command
    command = [
        "python", script_path,
        "--NAME_RESULTS_FOLDER", name_results_folder,
        "--PATH_INPUT_DATA", path_input_data,
        "--PATH_RESULTS_BASE", path_results_base,
        "--NUMBER_FOLDS", str(number_folds),
        "--NUMBER_REPETITIONS", str(number_repetit),
        "--CLASSIFIER", classifier,
        "--HP_TUNING", hp_tuning,
        "--CALC_SHAP_VALUES", SHAP
    ]
    try:
        output = subprocess.check_output(command, text=True)
        print(output)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")

    root.destroy()


# Create the main window
root = tk.Tk()
root.title("Run Python Script with Arguments")

# Input data path
label_path_input_data = Label(
    root, text="Select the folder containing the input data:")
label_path_input_data.pack(pady=5)
entry_path_input_data = Entry(root, width=50)
entry_path_input_data.pack(pady=5)
button_browse_input_data = Button(
    root, text="Browse", command=choose_input_data_path)
button_browse_input_data.pack(pady=5)

# Name results folder
label_name_results_folder = Label(
    root, text="Specify a name for the results folder:")
label_name_results_folder.pack(pady=5)
entry_name_results_folder = Entry(root, width=50)
entry_name_results_folder.pack(pady=5)

# Path results base
label_path_results_base = Label(
    root, text="Select a directory for the results folder:")
label_path_results_base.pack(pady=5)
entry_path_results_base = Entry(root, width=50)
entry_path_results_base.pack(pady=5)
button_browse_results_base = Button(
    root, text="Browse", command=choose_results_base_path)
button_browse_results_base.pack(pady=5)

# Number folds
label_number_folds = Label(
    root, text="Number of folds in the cross-validation")
label_number_folds.pack(pady=5)
number_folds_options = [5, 10]
number_folds_var = IntVar(root)
number_folds_var.set(number_folds_options[0])  # Set default value
option_menu_number_folds = OptionMenu(
    root, number_folds_var, *number_folds_options)
option_menu_number_folds.pack(pady=5)

# Number repeats of cross-validation
label_number_repetit = Label(
    root, text="Number of repetitions of the cross-validation")
label_number_repetit.pack(pady=5)
number_repetit_options = [1, 50, 100]
number_repetit_var = IntVar(root)
number_repetit_var.set(number_repetit_options[0])  # Set default value
option_menu_number_repetit = OptionMenu(
    root, number_repetit_var, *number_repetit_options)
option_menu_number_repetit.pack(pady=5)

# Classifier
label_classifier = Label(root, text="Which classifier do you want to use?")
label_classifier.pack(pady=5)
classifier_options = ["ridge_regression", "random_forest", "xgboost"]
classifier_var = StringVar(root)
classifier_var.set(classifier_options[0])  # Set default value
option_menu_classifier = OptionMenu(root, classifier_var, *classifier_options)
option_menu_classifier.pack(pady=5)

# Hp tuning
label_hp_tuning = Label(root, text="Should hyperparameter tuning be applied?:")
label_hp_tuning.pack(pady=5)
hp_tuning_options = ["False", "True"]
hp_tuning_var = StringVar(root)
hp_tuning_var.set(hp_tuning_options[0])  # Set default value
option_menu_hp_tuning = OptionMenu(root, hp_tuning_var, *hp_tuning_options)
option_menu_hp_tuning.pack(pady=5)

#  SHAP values
label_SHAP = Label(root, text="Should SHAP values be calculated?:")
label_SHAP.pack(pady=5)
SHAP_options = ["False", "True"]
SHAP_var = StringVar(root)
SHAP_var.set(SHAP_options[0])  # Set default value
option_menu_SHAP = OptionMenu(root, SHAP_var, *SHAP_options)
option_menu_SHAP.pack(pady=5)


button_run_script = tk.Button(root, text="Run Script", command=run_script)
button_run_script.pack(pady=20)

# Start the main event loop
root.mainloop()
