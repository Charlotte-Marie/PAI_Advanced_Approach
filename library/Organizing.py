# -*- coding: utf-8 -*-
"""
This script defines functions to organize the folder structure.

@author: Charlotte Meinke, Silvan Hornstein, Rebecca Delfendahl, Till Adam & Kevin Hilbert
"""

# %% Import functions
import sys
import os

# %%


def create_folder_to_save_results(PATH_RESULTS):
    """
    Create folder to save results

    Parameters:
    - PATH_RESULTS : str
      The path to the main results folder.

    Raises:
    - sys.exit
      If the results folder already exists, the script is aborted to prevent overwriting previous results.

    """
    # Create folder for result. The folder is named after different options (e.g., classifier used and whether hp tuning was applied)
    # Stop script, if the folder already exists, to prevent overwriting
    if not os.path.exists(os.path.join(PATH_RESULTS)):
        os.makedirs(PATH_RESULTS)
    elif os.path.exists(os.path.join(PATH_RESULTS)):
        raise ValueError(
            'Please use a new model name or delete existing analysis')


def get_categorical_variables(catvars_import_path):
    """
    Check if file is available specifying names of the categorical variables and return list with these variable names, otherwise return empty list

    Parameters:
    - catvars_import_path : str
      The path to the tab-delimited text file with the variable names.

    """
    # Try to import file specifying names of the categorical variables, get variable names from the file and return list with variable names
    # Return empty list, if the file does not exist, to identify categorical variable based on strings later on
    try:
        with open(catvars_import_path, 'r') as file:
            names_categorical_vars = file.readline().strip().split('\t')
        return names_categorical_vars
    except (FileNotFoundError):
        names_categorical_vars = []
        return names_categorical_vars
