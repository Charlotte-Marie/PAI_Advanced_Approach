from jinja2 import Template
import os

# Set paths
summary_path = "Euer Path zu: PAI_summary_all.txt"
model_performance_path = "Euer Path zu: modelperformance_summary.txt"
plots_directory = "Euer Path zu: ridge_regression/plots"

# Organizing files in "plots"
plot_files = os.listdir(plots_directory)
pai_distribution_files = [file for file in plot_files if file.startswith("PAI_distribution")]
optimal_vs_nonoptimal_files = [file for file in plot_files if file.startswith("optimal_vs_nonoptimal")]
# Sorting files in "plots"
pai_distribution_files.sort()
optimal_vs_nonoptimal_files.sort()
pai_distribution_files = [os.path.join(plots_directory, file) for file in pai_distribution_files]
optimal_vs_nonoptimal_files = [os.path.join(plots_directory, file) for file in optimal_vs_nonoptimal_files]

# Read the PAI summary file
with open(summary_path, 'r') as summary_file:
    lines = summary_file.readlines()

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
        model_performance_data['Mean_all_RMSE'],
        model_performance_data['Mean_option_A_RMSE'],
        model_performance_data['Mean_option_B_RMSE']
    ],
    'Mean_Cor': [
        model_performance_data['Mean_all_correlation'],
        model_performance_data['Mean_option_A_correlation'],
        model_performance_data['Mean_option_B_correlation']
    ],
    'Mean_MAE': [
        model_performance_data['Mean_all_MAE'],
        model_performance_data['Mean_option_A_MAE'],
        model_performance_data['Mean_option_B_MAE']
    ],
}

# Data for the html template
template_data = {
    'title': 'PAI Report',
    'main_heading': 'Report Personalized Advantage Index',
    'sub_header_1': 'Evaluating the PAI across 100 repetitions of the CV',
    'sub_header_2': 'Evaluating the PAI: Results per repetition',
    'sub_header_3': 'Model performance across 100 x 5-fold CV',
    'column_names': column_names,
    'summary_values': summary_values,
    'model_performance_metrics': model_performance_metrics,
#    'figure_1_title': 'Distribution of the absolute PAI',
#    'figure_2_title': 'Distribution of outcome optimal and nonoptimal',
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
