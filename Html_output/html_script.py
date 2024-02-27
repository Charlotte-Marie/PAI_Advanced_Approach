# -*- coding: utf-8 -*-
"""
"""

from jinja2 import Template

# Data for the template
template_data = {
    'title': 'Your Plots',
    'heading': 'Your Plots and Text',
    'plot_filenames': ["C:\\Users\\charl\\Documents\\Promotion\\results_synth\\random_forest\\plots\\optimal_vs_nonoptimal_0_all.png",
                       "C:\\Users\\charl\\Documents\\Promotion\\results_synth\\random_forest\\plots\\optimal_vs_nonoptimal_1_all.png"],
    'additional_text': 'Additional text goes here.',
}

# Read the template file
with open('html_template_click.html', 'r') as template_file:
    template_content = template_file.read()

# Create a Jinja2 template
template = Template(template_content)

# Render the template with data
html_content = template.render(template_data)

# Save HTML content to a file
with open('output.html', 'w') as f:
    f.write(html_content)
