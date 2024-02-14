# -*- coding: utf-8 -*-
"""
This script generates a graphical user interface to run the script: Main_PAI_advanced_approach.py

@author: meinkcha
"""

import tkinter as tk
from tkinter import Entry, Label, OptionMenu, IntVar, StringVar, Button, filedialog
from tkinter import ttk
import subprocess
import threading
import time

def choose_input_data_path():
    path = filedialog.askdirectory()
    entry_path_input_data.delete(0, tk.END)
    entry_path_input_data.insert(0, path)
    
def choose_results_base_path():
    path = filedialog.askdirectory()
    entry_path_results_base.delete(0, tk.END)
    entry_path_results_base.insert(0, path)
    
def long_running_function():
    # Simulate a long-running task
    time.sleep(5)
    print("Task completed!")

    # After the task is completed, close the GUI
    root.destroy()

def start_task():
    # Disable the button to prevent multiple clicks
    start_button['state'] = 'disabled'

    # Start the long-running task in a separate thread
    thread = threading.Thread(target=long_running_function)
    thread.start()

def run_script():
    # Hardcoded script path
    script_path = "Main_PAI_advanced_approach.py"
    
    # Get user-provided arguments from entry widgets
    path_input_data = entry_path_input_data.get()
    input_data_name = entry_input_data_name.get()
    path_results_base = entry_path_results_base.get()
    # number_folds = number_folds_var.get()
    # number_repeats = number_repeats_var.get()
    classifier = classifier_var.get()
    hp_tuning = hp_tuning_var.get()

    # Construct the command
    command = [
        "python", script_path,
        f"--PATH_INPUT_DATA={path_input_data}",
        f"--INPUT_DATA_NAME={input_data_name}",
        f"--PATH_RESULTS_BASE={path_results_base}",
        # f"--NUMBER_FOLDS={number_folds}",
        # f"--NUMBER_REPEATS={number_repeats}",
        f"--CLASSIFIER={classifier}",
        f"--HP_TUNING={hp_tuning}"
    ]

    subprocess.run(command)

# Create the main window
root = tk.Tk()
root.title("Run Python Script with Arguments")

# Create and pack widgets
label_path_input_data = Label(root, text="Choose folder with input data:")
label_path_input_data.pack(pady=5)
entry_path_input_data = Entry(root, width=50)
entry_path_input_data.pack(pady=5)
button_browse_input_data = Button(root, text="Browse", command=choose_input_data_path)
button_browse_input_data.pack(pady=5)

label_input_data_name = Label(root, text="Name of input dataset:")
label_input_data_name.pack(pady=5)
entry_input_data_name = Entry(root, width=50)
entry_input_data_name.pack(pady=5)

label_path_results_base = Label(root, text="Choose folder that will contain the results-folder:")
label_path_results_base.pack(pady=5)
entry_path_results_base = Entry(root, width=50)
entry_path_results_base.pack(pady=5)
button_browse_results_base = Button(root, text="Browse", command=choose_results_base_path)
button_browse_results_base.pack(pady=5)

label_classifier = Label(root, text="Which classifier do you want to use?")
label_classifier.pack(pady=5)
classifier_options = ["ridge_regression", "random_forest"]
classifier_var = StringVar(root)
classifier_var.set(classifier_options[0])  # Set default value
option_menu_classifier = OptionMenu(root, classifier_var, *classifier_options)
option_menu_classifier.pack(pady=5)

# label_number_folds = Label(root, text="Number of folds in the cross-validation")
# label_number_folds.pack(pady=5)
# number_folds_options = [5,10]
# number_folds_var = IntVar(root)
# number_folds_var.set(number_folds_options[0])  # Set default value
# option_menu_number_folds = OptionMenu(root, number_folds_var, *number_folds_options)
# option_menu_number_folds.pack(pady=5)

# label_number_repeats = Label(root, text="Which number_repeats do you want to use?")
# label_number_repeats.pack(pady=5)
# number_repeats_options = [1,50,100]
# number_repeats_var = IntVar(root)
# number_repeats_var.set(number_repeats_options[0])  # Set default value
# option_menu_number_repeats = OptionMenu(root, number_repeats_var, *number_repeats_options)
# option_menu_number_repeats.pack(pady=5)

label_hp_tuning = Label(root, text="Should hyperparameter tuning be applied?:")
label_hp_tuning.pack(pady=5)
hp_tuning_options = ["False","True"]
hp_tuning_var = StringVar(root)
hp_tuning_var.set(hp_tuning_options[0])  # Set default value
option_menu_hp_tuning = OptionMenu(root, hp_tuning_var, *hp_tuning_options)
option_menu_hp_tuning.pack(pady=5)

label = ttk.Label(root, text="Click the button to start the script")
label.pack(padx=10, pady=10)

start_button = ttk.Button(root, text="Start Task", command=start_task)
start_button.pack(padx=10, pady=10)

# Start the main event loop
root.mainloop()

