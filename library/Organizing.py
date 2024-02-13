# -*- coding: utf-8 -*-
"""
This script defines functions to organize the folder structure.

@author: charl
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
        print('Please use a new model name or delete existing analysis')
        sys.exit("Execution stopped")
