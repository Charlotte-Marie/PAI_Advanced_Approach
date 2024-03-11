from jinja2 import Template
import os
import re

# Set paths
summary_path = "Euer Path zu: PAI_summary_all.txt"
model_performance_path = "Euer Path zu: modelperformance_summary.txt"
plots_directory = "Euer Path zu: ridge_regression/plots"

#Extracting numbers from filenames in plots
def extract_number(filename):
    matches = re.search(r"(\d+)", filename)
    return int(matches.group(1)) if matches else 0

# Organizing and sorting files in "plots"
plot_files = os.listdir(plots_directory)
pai_distribution_files = [file for file in plot_files if file.startswith("PAI_distribution") and file.endswith("_all.png")]
optimal_vs_nonoptimal_files = [file for file in plot_files if file.startswith("optimal_vs_nonoptimal") and file.endswith("_all.png")]

# Interpreting numbers in plot images as integers instread of strings
pai_distribution_files.sort(key=extract_number)
optimal_vs_nonoptimal_files.sort(key=extract_number)

# Prepend the full path for accessing the images (if necessary)
pai_distribution_files = [os.path.join(plots_directory, file) for file in pai_distribution_files]
optimal_vs_nonoptimal_files = [os.path.join(plots_directory, file) for file in optimal_vs_nonoptimal_files]

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
summary_values = lines[1].strip().split('\t')[1:] # skip first value (row index)

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
        round(model_performance_data['Mean_all_RMSE'],2),
        round(model_performance_data['Mean_option_A_RMSE'],2),
        round(model_performance_data['Mean_option_B_RMSE'],2)
    ],
    'Mean_Cor': [
        round(model_performance_data['Mean_all_correlation'],2),
        round(model_performance_data['Mean_option_A_correlation'],2),
        round(model_performance_data['Mean_option_B_correlation'],2)
    ],
    'Mean_MAE': [
        round(model_performance_data['Mean_all_MAE'],2),
        round(model_performance_data['Mean_option_A_MAE'],2),
        round(model_performance_data['Mean_option_B_MAE'],2)
    ],
}

# Data for the html template
template_data = {
    'title': 'PAI Report',
    'main_heading': 'Report Personalized Advantage Index',
    'sub_header_1': 'Evaluating the PAI across 100 repetitions of the CV',
    'sub_header_2': 'Evaluating the PAI: Results per repetition',
    'sub_header_3': 'Model performance across 100 x 5-fold CV',
    'n_sig_t_test': n_sig_t_test,
    'mean_cohens_d': mean_cohens_d,
    'mean_cohens_d_sd': mean_cohens_d_sd,
    'n_variance_homogeneity_violated': n_variance_homogeneity_violated,
    'n_normality_assumption_violated': n_normality_assumption_violated,
    'model_performance_metrics': model_performance_metrics,
    'plot_filenames': {
        'pai_distribution': pai_distribution_files,
        'optimal_vs_nonoptimal': optimal_vs_nonoptimal_files,
    },
    'additional_text': 'Additional text goes here.',
}


with open('html_template_click_RD.html', 'r') as template_file:
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

# Save HTML content to a file
with open('output.html', 'w') as f:
    f.write(html_content)
