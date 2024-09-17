"""
The script creates a nice htlm-output from the results of the script "Main_PAI_advanced_approach.py"

@author: Till Adam, Rebecca Delfendahl, Charlotte Meinke, Silvan Hornstein & Kevin Hilbert
"""

# %% Import packages
from jinja2 import Template
import os
import re

# %% Functions


def PAI_to_HTML(PATH_RESULTS, plots_directory, number_folds, number_repetit):
    """ This function creates a nice htlm-output from the results of the script "Main_PAI_advanced_approach.py"

    Parameters:
    - PATH_RESULTS: Directory with PAI_summary_all.txt and modelperformance_summary.txt
    - plots_directory: Directory containing the plots PAI_distribution and optimal vs. nonoptimal
    - number_folds: Number of folds in the cross-validation
    - number_repetit: Number of repetititons of the cross-validation

    """

    # Get specific paths
    summary_path = os.path.join(PATH_RESULTS, "PAI_summary_all.txt")
    model_performance_path = os.path.join(
        PATH_RESULTS, "modelperformance_summary.txt")

    # Extracting numbers from filenames in plots
    def extract_number(filename):
        matches = re.search(r"(\d+)", filename)
        return int(matches.group(1)) if matches else 0

    # Organizing and sorting files in "plots"
    plot_files = os.listdir(plots_directory)
    pai_distribution_files = [file for file in plot_files if file.startswith(
        "PAI_distribution") and file.endswith("_all.png")]
    optimal_vs_nonoptimal_files = [file for file in plot_files if file.startswith(
        "optimal_vs_nonoptimal") and file.endswith("_all.png")]
    SHAP_beeswarm_files = [file for file in plot_files if file.startswith(
        "Shap_Beeswarm")]

    # Interpreting numbers in plot images as integers instead of strings
    pai_distribution_files.sort(key=extract_number)
    optimal_vs_nonoptimal_files.sort(key=extract_number)
    SHAP_beeswarm_files.sort(key=extract_number)

    # Prepend the full path for accessing the images (if necessary)
    pai_distribution_files = [os.path.join(
        plots_directory, file) for file in pai_distribution_files]
    optimal_vs_nonoptimal_files = [os.path.join(
        plots_directory, file) for file in optimal_vs_nonoptimal_files]
    SHAP_beeswarm_files = [os.path.join(
        plots_directory, file) for file in SHAP_beeswarm_files]

    # Read the PAI summary file
    with open(summary_path, 'r') as summary_file:
        lines = summary_file.readlines()
    n_sig_t_test = lines[1].strip().split('\t')[1]
    mean_cohens_d = round(float(lines[1].strip().split('\t')[4]), 2)
    mean_cohens_d_sd = round(float(lines[1].strip().split('\t')[5]), 2)
    n_variance_homogeneity_violated = lines[1].strip().split('\t')[6]
    n_normality_assumption_violated = lines[1].strip().split('\t')[7]

    # Extract column names and values
    column_names = lines[0].strip().split('\t')
    summary_values = lines[1].strip().split(
        '\t')[1:]  # skip first value (row index)

    # Read and process the model performance summary file
    model_performance_data = {}
    with open(model_performance_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')  # Split the line by tab
            if len(parts) == 2:  # Check if the line contains exactly two values
                key, value = parts
                model_performance_data[key] = float(value)

    # Extracting specific metrics for the HTML template
    model_performance_metrics = {
        'Mean_MSE': [
            round(model_performance_data['Mean_all_RMSE'], 2),
            round(model_performance_data['Mean_option_A_RMSE'], 2),
            round(model_performance_data['Mean_option_B_RMSE'], 2)
        ],
        'Mean_Cor': [
            round(model_performance_data['Mean_all_correlation'], 2),
            round(model_performance_data['Mean_option_A_correlation'], 2),
            round(model_performance_data['Mean_option_B_correlation'], 2)
        ],
        'Mean_MAE': [
            round(model_performance_data['Mean_all_MAE'], 2),
            round(model_performance_data['Mean_option_A_MAE'], 2),
            round(model_performance_data['Mean_option_B_MAE'], 2)
        ],
    }

    # Data for the html template
    template_data = {
        'title': 'PAI Report',
        'main_heading': 'Report Personalized Advantage Index',
        'sub_header_1': f'Evaluating the PAI: Results across {number_repetit} repetitions of {number_folds}-fold cross-validation',
        'explanation_1': "To evaluate the potential usefulness of the PAI for treatment selection, post-treatment severity values are compared between patients who received their optimal treatment according to the PAI recommendation and those who received their non-optimal treatment. \nHere, an independent one-sided t-test is conducted, testing whether post-treatment severity scores of patients who received their optimal treatment were smaller than those of patients who received their nonoptimal treatment. \nThis is done for each repetition. Therefore, results across repetitions are summarized by reporting the number of significant t-test and mean Cohen's d. Furthermore, the number of repetitions in which assumptions of the t-test are violated is reported. In case of strong violations, alternatives such as the Welch-t-test (implemented here) should be used.",
        'sub_header_2': 'Evaluating the PAI: Plots per repetition',
        'explanation_2': 'Please click to receive the plots for the next repetition. In the right plot, the horizontal dashed lines represent the mean post-treatment severity for each group.',
        'sub_header_3': f'Evaluating the model performance across {number_repetit} x {number_folds}-fold cross-validation',
        'explanation_3': 'The performance of the underlying models used in the PAI is presented, both across separate models for treatments and for each model seperately',
        'sub_header_4': 'SHAP values if calculated',
        'n_sig_t_test': n_sig_t_test,
        'mean_cohens_d': mean_cohens_d,
        'mean_cohens_d_sd': mean_cohens_d_sd,
        'n_variance_homogeneity_violated': n_variance_homogeneity_violated,
        'n_normality_assumption_violated': n_normality_assumption_violated,
        'model_performance_metrics': model_performance_metrics,
        'plot_filenames': {
            'pai_distribution': pai_distribution_files,
            'optimal_vs_nonoptimal': optimal_vs_nonoptimal_files,
            'SHAP': SHAP_beeswarm_files,
        }
    }

    script_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(script_dir, 'html_template_click_RD.html'), 'r') as template_file:
        template_content = template_file.read()

    # Create a Jinja2 template
    template = Template(template_content)

    # Render the template with data
    html_content = template.render(template_data)

    # Save HTML content to a file
    with open('output.html', 'w') as f:
        f.write(html_content)

    # Save HTML content to a file
    with open('output.html', 'w') as f:
        f.write(html_content)


if __name__ == "__main__":
    # Define the paths as arguments to be passed to the function
    summary_path = "Your path to: PAI_summary_all.txt"
    model_performance_path = "Your path to: modelperformance_summary.txt"
    plots_directory = "Your path to: ridge_regression/plots"

    # Now, call the function with the paths as arguments
    PAI_to_HTML(summary_path, model_performance_path, plots_directory)
