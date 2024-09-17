# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:35:04 2024

@author: Charlotte Meinke, Silvan Hornstein, Rebecca Delfendahl, Till Adam & Kevin Hilbert
"""

# %% Import packages
import unittest
import pandas as pd
import numpy as np

import os 
import sys
PROJECT_PATH = os.path.dirname(os.getcwd())
LIB_PATH = os.path.join(PROJECT_PATH,"library")
sys.path.append(LIB_PATH)

from Imputing import MiceModeImputer_pipe

# %% Test classes
class TestMiceModeImputer(unittest.TestCase):
    
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'binary_var': [0.5, -0.5, np.nan, 0.5],
            'categorical_var': ['cat', 'dog',np.nan, 'cat'],
            'dimensional_var': [1.0, 2.1, np.nan, 4.5],
            'too_many_miss': [np.nan, 1, np.nan, np.nan]
        }
        
        self.df = pd.DataFrame(data)
        self.cat_vars = ['categorical_var']
        
    def test_transform_w_catvars_as_strings(self):
        # Initialize MiceModeImputer_pipe instance
        mice_imputer = MiceModeImputer_pipe()
        
        # Fit and transform the sample data
        transformed_data = mice_imputer.fit_transform(self.df, cat=self.cat_vars)
        
        # Check if the transformation has the expected number of columns
        # 1 binary, 1 one-hot encoded categorical, 1 dimensional = 3 columns
        # + (binary_var gets recoded into -0.5, 0.5)
        self.assertEqual(transformed_data.shape[1], 3)
        
        # Check the values of the transformed data
        expected_binary = np.array([0.5, -0.5, 0.5, 0.5])
        expected_categorical = np.array([-0.5, 0.5, -0.5, -0.5])
        
        self.assertTrue(np.allclose(transformed_data[:, 0], expected_binary, equal_nan=True))
        self.assertTrue(np.allclose(transformed_data[:, 1], expected_categorical, equal_nan=True))
        
    def test_transform_wout_catvars_as_strings(self):
         # Initialize MiceModeImputer_pipe instance
         mice_imputer = MiceModeImputer_pipe()
         
         # Fit and transform the sample data
         transformed_data = mice_imputer.fit_transform(self.df)
         
         # Check if the transformation has the expected number of columns
         # 1 binary, 1 one-hot encoded categorical, 1 dimensional = 3 columns
         # + (binary_var gets recoded into -0.5, 0.5)
         self.assertEqual(transformed_data.shape[1], 3)
         
         # Check the values of the transformed data
         expected_binary = np.array([0.5, -0.5, 0.5, 0.5])
         expected_categorical = np.array([-0.5, 0.5, -0.5, -0.5])
         
         self.assertTrue(np.allclose(transformed_data[:, 0], expected_binary, equal_nan=True))
         self.assertTrue(np.allclose(transformed_data[:, 1], expected_categorical, equal_nan=True))
    
    def test_get_feature_names(self):
        # Initialize MiceModeImputer_pipe instance
        mice_imputer = MiceModeImputer_pipe()
        
        # Fit and transform the data
        mice_imputer.fit(self.df, cat_vars=self.cat_vars)
        
        # Check the feature names (Dummy encoding for categorical variables!)
        expected_feature_names = ['binary_var', 'categorical_var_dog', 'dimensional_var']
        self.assertListEqual(mice_imputer.new_feat_names.tolist(), expected_feature_names)
        
    # def test_remove_columns_with_too_many_missing(self):
        # Initialize MiceModeImputer_pipe instance
        mice_imputer = MiceModeImputer_pipe()
        
        # Fit and transform the data
        transformed_data = mice_imputer.fit_transform(self.df, cat=self.cat_vars)
        
        # Check that the 'too_many_miss' column is removed
        self.assertNotIn('too_many_miss', mice_imputer.new_feat_names)

# %% Main
if __name__ == '__main__':
    unittest.main()