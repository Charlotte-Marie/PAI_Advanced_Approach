# PAI

The Personalized Advantage Index (PAI) was introduced first by DeRubeis et al. (2014; see References) and is a well-used approach to predict which one of several available treatment alternatives is optimal for an individual patient. The PAI has seen many different implementations. Here, we provide a low-bias pipeline for use by the scientific community.

# Prepare the Script

1. Clone the repository
2. Make sure all needed requirements for this script are installed. 
    
    a) For conda-users, run:
    
    ```python
    
    ```
    
     b) For non-conda-users, run:
    
    ```python
     pip install -r "requirements.txt".
    ```
    

# Run the script

There are two ways to run the script. In both ways, the following arguments need to be given:

- PATH_INPUT_DATA → The path to the folder which contains features.txt, labels.txt, and groups.txt
- INPUT_DATA_NAME → Name of input data or more general string that will be part of the name of the results folder
- CLASSIFIER → Choose ridge_regression OR random_forest
- HP_TUNING → Choose True OR False

## Run script via Terminal

a) Make sure that the section for running the script via IDE is commented

```python
 		# Run script via IDE (start)
    # PATH_RESULTS, PATH_RESULTS_PLOTS, PATH_INPUT_DATA, OPTIONS = set_paths_and_options(
    #     PATH_INPUT_DATA=path_panik,
    #     INPUT_DATA_NAME="PANIK",
    #     CLASSIFIER="ridge_regression",  # ridge_regression OR random_forest
    #     HP_TUNING="False")
    # Run script via IDE (end)
```

b) Open a terminal window.

c) Navigate to the directory where your script is located.

d) Run the following command:

```python
Main_PAI_advanced_approach.py --PATH_INPUT_DATA *Your_path -*-INPUT_DATA_NAME *Your_data_name*
```

## Run script in your IDE (e.g., Spyder)

1. Make sure that the section for running the script via terminal is commented

```python
		# Run script via terminal (start)
    # parser = argparse.ArgumentParser(
    #     description='Advanced script to calculate the PAI')
    # parser.add_argument('--PATH_INPUT_DATA', type=str,
    #                     help='Path to input data')
    # parser.add_argument('--INPUT_DATA_NAME', type=str,
    #                     help='Name of input dataset')
    # parser.add_argument('--CLASSIFIER', type=str,
    #                     help='Classifier to use, set ridge_regression or random_forest')
    # parser.add_argument('--HP_TUNING', type=str,
    #                     help='Should hyperparameter tuning be applied? Set False or True')
    # args = parser.parse_args()

    # PATH_RESULTS, PATH_RESULTS_PLOTS, PATH_INPUT_DATA, OPTIONS = set_paths_and_options(PATH_INPUT_DATA=args.PATH_INPUT_DATA,
    #                                                                                    INPUT_DATA_NAME=args.INPUT_DATA_NAME,
    #                                                                                    CLASSIFIER=args.CLASSIFIER,
    #                                                                                    HP_TUNING=args.HP_TUNING)
    # Run script via terminal (end)
```

2. Set the argument needed in the script

```python
 PATH_RESULTS, PATH_RESULTS_PLOTS, PATH_INPUT_DATA, OPTIONS = set_paths_and_options(
         PATH_INPUT_DATA= "Your_path",
         INPUT_DATA_NAME="Your_name",
         CLASSIFIER="ridge_regression",  # ridge_regression OR random_forest
         HP_TUNING="False") # True or False
```

3. Run the script

# Additional Customization

Besides these main arguments that need to be set, you easily change several settings within the function set_paths_and_options. The most important ones are:

```python
OPTIONS['number_folds'] = 5 # number of folds in the cross-validation
OPTIONS["number_repeats"] = 100 # number of repetitions of the cross-validation
```

# Prepare Input Data

Before running the script, make sure you have three files ready: features.txt, labels.txt, and groups.txt. These files should contain information about features, post-treatment severity, and group membership for all patients.

General requirements:

- Put all three files in one folder (and specify the folder path as INPUT_DATA_PATH)
- Make sure that all files are tab-delimited text files (.txt). Usually, tab-delimited text can easily be exported from statistic programs or excel.
- Make sure that a period is used as a decimal separator and avoid special characters in variable names.
- Make sure that variable names are on the top line.

Specific requirements:

features.txt: Certain processes in the script hinge on distinguishing feature types (binary, categorical, or dimensional). To facilitate the detection of feature type, recode binary variables as 0.5 and -0.5 and provide categorial variables in string format.

groups.txt: Make sure that it is a binary variable. The format does not matter (string or numerical).

Address missing values:

- Do not impute missing values in advance, as this could lead to data leakage.
- The script autonomously handles imputation for missing values marked as NA. Ensure that missing values are denoted appropriately.

# Empirical and theoretical foundations of design choices

There are plenty of different options for preparing the data and the machine learning pipeline. Mostly, no clear data is available suggesting which approches are superior to others. Still, there were some papers that we considered important when designing this pipeline, which are presented below:

- Centering of variables and centering of binary variables to -0.5 and 0.5 -- Kraemer & Blasey (2004)
- Elastic net feature reduction -- Bennemann et al. (2022)
- Random Forest -- Grinsztajn et al (preprint)
- Refraining from LOO-CV and using a repeated 5-fold stratified train-test split -- Varoquaux (2018) & Varoquaux et al. (2017) & Flint et al. (2021) & the observation that prediction performance varies substantially between iterations in our own previous papers, including Leehr et al. (2021) & Hilbert et al. (2021)

# **References**

1. Bennemann et al. (2022). Predicting patients who will drop out of out-patient psychotherapy using machine learning algorithms. The British Journal of Psychiatry, 220, 192–201.
2. DeRubeis et al (2014). The Personalized Advantage Index: translating research on prediction into individualized treatment recommendations. A demonstration. PLOS One, 9(1), e83875.
3. Flint et al. (2021). Systematic misestimation of machine learning performance in neuroimaging studies of depression. Neuropsychopharmacology, 46, 1510–1517.
4. Grinsztajn et al (preprint). Why do tree-based models still outperform deep learning on tabular data? arXiv, 2207.08815.
5. Hilbert et al. (2021). Identifying CBT non-response among OCD outpatients: a machine-learning approach. Psychotherapy Research, 31(1), 52-62.
6. Kraemer & Blasey (2004). Centring in regression analyses: a strategy to prevent errors in statistical inference. International Journal of Methods in Psychiatric Research, 13(3), 141-51.
7. Leehr et al. (2021). Clinical predictors of treatment response towards exposure therapy in virtuo in spider phobia: a machine learning and external cross-validation approach. Journal of Anxiety Disorders, 83, 102448.
8. Varoquaux (2018). Cross-validation failure: Small sample sizes lead to large error bars. NeuroImage, 180(A), 68-77.
9. Varoquaux et al. (2017). Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines. NeuroImage, 145, 166–179.
