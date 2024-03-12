# PAI

The Personalized Advantage Index (PAI) was introduced first by DeRubeis et al. (2014; see References) and is a well-used approach to predict which one of several available treatment alternatives is optimal for an individual patient. The PAI has seen many different implementations. Here, we provide a low-bias pipeline for use by the scientific community.

# Getting Started

Follow these steps to quickly set up the program and analyze data using the provided example:

1. **Clone this repository** to your local machine.
2. Make sure you have installed the packages numpy, pandas, and scikit.
3. **Open and run the script "[GUI.py](http://gui.py/)".**
    1. *Folder with input data*: select "synthet_test_data" folder for using example data.
    2. Select name and location of results-folder. Leave analysis parameters at default.
4. **View results**: after running the script, plots and summary statistics can be found in the specified results-folder

# Preparation

1. Clone the repository
   
   By cloning the repository, you replicate the entire repository, including all its files and folders, maintaining the same structure on your local computer. If you are not experienced in working with Github, we recommend to use the Github Desktop App: https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop.

   Alternatively, you have the option to manually download the required files and reconstruct the necessary folder structure. If you choose this method, please ensure that you download the following key components: **Main_PAI_advanced_approach.py**, **GUI.py**, the **library** folder, and, if you wish to experiment with the synthetically generated test data, also download the **synth_test_data** folder.
   
2. Make sure all needed requirements are installed.

   a) Manually

      You only need three additional packages (numpy, pandas, scikit). Install them manually if you have not installed them yet.
    
    b) Automatically for conda-users, run in the terminal:
    
    ```python
    conda env create -f "YOUR_PATH_TO_THE_ENVIRONMENT\Environment.yaml"
    ```
    If you have not changed the location of the .yaml file after cloning the repository, *YOUR_PATH_TO_THE_ENVIRONMENT* equals *YOUR_PATH_TO_THE_REPOSITORY*
    
    c) Automatically for non-conda-users, run in the terminal:
    
    ```python
    pip install -r "requirements.txt".
    ```
    

# Run the Main script
To calculate the PAI with our low bias approach, the script **"Main_PAI_advanced_approach.py"** needs to be executed.
To execute the script, there are three methods available, each requiring the same set of arguments to be provided:

- PATH_INPUT_DATA → The path to the folder which contains features.txt, labels.txt, and groups.txt
- INPUT_DATA_NAME → Name of input data or more general string that will be part of the name of the results folder
- RESULTS_PATH_BASE → Path to store the results folder
- NUMBER_FOLDS → Number of folds in the cross-validation (typically 5 or 10)
- NUMBER_ITERIT → Number of repetitions of the cross-validation (typically 100, choose 1 for a first try)
- CLASSIFIER → Choose ridge_regression OR random_forest
- HP_TUNING → Choose True OR False

## 1. Run script via Graphical User Interface (most simple)
Run the script "GUI.py". A small graphical user interface will launch automatically, allowing you to provide the arguments specified above as input.

## 2. Run script via (anaconda) terminal
a) Make sure that the section for running the script via IDE is commented in the script "Main_PAI_advanced_approach.py"

```python
    # Run script via IDE (start)
    # working_directory = os.getcwd()
    # path_data = os.path.join(working_directory, "synthet_test_data")
    # path_results_base = working_directory
    # PATH_INPUT_DATA = path_data
    # OPTIONS = set_options(classifier = "random_forest",
    #                       number_folds = 5,
    #                       number_repetit = 1,
    #                       hp_tuning = "false"
    #                       )
    # PATH_RESULTS = generate_and_create_results_path(path_results_base,
    #                                                 input_data_name = "test_data",
    #                                                 OPTIONS = OPTIONS)
    # Run script via IDE (end)
```

b) Open the terminal window in anaconda

c) Run the following command (Replace YOUR_PATH and YOUR_RESULTS_PATH with your own paths)

```python
python "YOUR_PATH\Main_PAI_advanced_approach.py" --INPUT_DATA_NAME test --PATH_INPUT_DATA "YOUR_PATH/PAI_Advanced_Approach/synthet_test_data" --PATH_RESULTS_BASE "YOUR_RESULTS_PATH" --NUMBER_FOLDS 5 --NUMBER_REPETIT 1 --CLASSIFIER ridge_regression --HP_TUNING False
```

## 3. Run script in your IDE (e.g., Spyder)

a) Make sure that the section for running the script via terminal is commented in the script "Main_PAI_advanced_approach.py".

```python
    # Run script via terminal or GUI (start)
    # parser = argparse.ArgumentParser(
    #     description='Advanced script to calculate the PAI')
    # parser.add_argument('--PATH_INPUT_DATA', type=str,
    #                     help='Path to input data')
    # parser.add_argument('--INPUT_DATA_NAME', type=str,
    #                     help='Name of input dataset')
    # parser.add_argument('--PATH_RESULTS_BASE', type=str,
    #                     help='Path to save results')
    # parser.add_argument('--NUMBER_FOLDS', type=int,
    #                     help='Number of folds in the cross-validation')
    # parser.add_argument('--NUMBER_REPETIT', type=int,
    #                     help='Number of repetitions of the cross-validation')
    # parser.add_argument('--CLASSIFIER', type=str,
    #                     help='Classifier to use, set ridge_regression or random_forest')
    # parser.add_argument('--HP_TUNING', type=str,
    #                     help='Should hyperparameter tuning be applied? Set False or True')
    # args = parser.parse_args()

    # PATH_INPUT_DATA = args.PATH_INPUT_DATA
    # OPTIONS = set_options(classifier=args.CLASSIFIER,
    #                       number_folds=args.NUMBER_FOLDS,
    #                       number_repetit=args.NUMBER_REPETIT,
    #                       hp_tuning=args.HP_TUNING
    #                       )
    # PATH_RESULTS = generate_and_create_results_path(path_results_base=args.PATH_RESULTS_BASE,
    #                                                 input_data_name=args.INPUT_DATA_NAME,
    #                                                 OPTIONS=OPTIONS)
    # Run script via terminal or GUI (end)
```

b) Set the arguments needed in the script

```python
    working_directory = os.getcwd()
    path_data = os.path.join(working_directory, "synthet_test_data")
    path_results_base = working_directory
    PATH_INPUT_DATA = path_data
    OPTIONS = set_options(classifier = "random_forest",
                          number_folds = 5,
                          number_repetit = 1,
                          hp_tuning = "false"
                          )
    PATH_RESULTS = generate_and_create_results_path(path_results_base,
                                                    input_data_name = "test_data",
                                                    OPTIONS = OPTIONS)
```

c) Run the script

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
