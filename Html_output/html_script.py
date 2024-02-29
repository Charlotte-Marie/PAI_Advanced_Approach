from jinja2 import Template

# Set paths
summary_path = "C:/Users/tillj/Documents/GitHub/PAI_Advanced_Approach/Html_output/results_test_data/ridge_regression/PAI_summary_all.txt"
model_performance_path = "C:/Users/tillj/Documents/GitHub/PAI_Advanced_Approach/Html_output/results_test_data/ridge_regression/modelperformance_summary.txt"

# Read the PAI summary file
with open(summary_path, 'r') as summary_file:
    lines = summary_file.readlines()

# Extract column names and values
column_names = lines[0].strip().split('\t')
summary_values = lines[1].strip().split('\t')[1:]

# Read and process the model performance summary file
model_performance_data = {}
with open(model_performance_path, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            key, value = parts
            model_performance_data[key] = float(value)

# Model performance metrics for HTML template
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
    'figure_1_title': 'Distribution of the absolute PAI',
    'figure_2_title': 'Distribution of outcome optimal and nonoptimal',
    #'plot_filenames': ["C:\\Users\\charl\\Documents\\Promotion\\results_synth\\random_forest\\plots\\optimal_vs_nonoptimal_0_all.png",
    #                   "C:\\Users\\charl\\Documents\\Promotion\\results_synth\\random_forest\\plots\\optimal_vs_nonoptimal_1_all.png"],
    'plot_filenames': ["C:\\Users\\Acer\\Documents\\GitHub\\PAI_Advanced_Approach\\Html_output\\results_test_data\\ridge_regression\\plots\\PAI_distribution_0_all.png",
                       "C:\\Users\\Acer\\Documents\\GitHub\\PAI_Advanced_Approach\\Html_output\\results_test_data\\ridge_regression\\plots\\optimal_vs_nonoptimal_0_all.png"], 
    'additional_text': 'Additional text goes here.',
}

# Read the template file
with open('html_template_click_RD.html', 'r') as template_file:
    template_content = template_file.read()

# Create a Jinja2 template
template = Template(template_content)

# Render the template with data
html_content = template.render(template_data)

# Save HTML content to a file
with open('output.html', 'w') as f:
    f.write(html_content)
