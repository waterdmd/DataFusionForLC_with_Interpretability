
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the analysis results
input_file = 'gradcam_results/all_gradcam_analysis.json'
with open(input_file, 'r') as f:
    all_gradcam_analysis = json.load(f)

# Identify all possible class labels
all_classes = set()
for gradcam_analysis in all_gradcam_analysis.values():
    for target_class in gradcam_analysis.keys():
        all_classes.add(int(target_class))
        for class_label in gradcam_analysis[target_class]['positive_counts'].keys():
            all_classes.add(int(class_label))

all_classes = sorted(all_classes)

# Initialize dictionaries to store the aggregated results
total_positive_counts = {target_class: {class_label: 0 for class_label in all_classes} for target_class in all_classes}
total_negative_counts = {target_class: {class_label: 0 for class_label in all_classes} for target_class in all_classes}

# Sum the counts for all images
for image_id, gradcam_analysis in all_gradcam_analysis.items():
    for target_class, counts in gradcam_analysis.items():
        target_class = int(target_class)
        for class_label, count in counts['positive_counts'].items():
            total_positive_counts[target_class][int(class_label)] += count
        for class_label, count in counts['negative_counts'].items():
            total_negative_counts[target_class][int(class_label)] += count

# Normalize the counts
def normalize_counts(counts):
    max_count = max(counts.values())
    return {class_label: count / max_count for class_label, count in counts.items()}

normalized_positive_counts = {target_class: normalize_counts(counts) for target_class, counts in total_positive_counts.items()}
normalized_negative_counts = {target_class: normalize_counts(counts) for target_class, counts in total_negative_counts.items()}

# Create output directory for saving plots
output_dir = 'gradcam_results/bar_plots'
os.makedirs(output_dir, exist_ok=True)

# Function to plot bar charts

def plot_bar_charts(pos_counts, neg_counts, class_labels, title, filename):
    x = np.arange(len(class_labels))
    pos_values = [pos_counts.get(int(class_label), 0) for class_label in class_labels]
    neg_values = [-neg_counts.get(int(class_label), 0) for class_label in class_labels]  # Make negative values for negative impact

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10), sharey=True)

    # Bar plot for positive and negative counts
    ax1.barh(x, pos_values, color='royalblue', label='Class Positive Impact')
    ax1.barh(x, neg_values, color='indianred', label='Class Negative Impact' )
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set_yticklabels(labels = "",fontsize = 12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_xlabel('Impact', fontsize = 12)
    ax1.legend(fontsize = 12)
    ax1.set_title(f'{title} - Positive and Negative Impact')

    # Bar plot for the difference
    difference = np.array(pos_values) + np.array(neg_values)  # Since neg_values are negative
    # Color based on the value of the difference
    colors = ['royalblue' if val > 0 else 'indianred' for val in difference]
    ax2.barh(x, difference, color=colors, label='Class Impact')
    ax2.set_yticklabels(labels = "",fontsize = 12)
    ax2.tick_params(axis='x', labelsize=12)
    
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Cum. Impact', fontsize = 12)
    ax2.legend(labelcolor = 'k',facecolor = 'w',fontsize = 12)
    ax2.set_title(f'{title} - Cumulative Impact')

    plt.yticks(x, class_labels)
    plt.ylabel('Class Labels', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()






# Plot bar charts for each target class
for target_class in total_positive_counts.keys():
    class_labels = [str(c) for c in all_classes]
    pos_counts = normalized_positive_counts[target_class]
    neg_counts = normalized_negative_counts[target_class]

    plot_bar_charts(pos_counts, neg_counts, class_labels,
                    f'Class-wise Impact for Class {target_class}', os.path.join(output_dir, f'bar_plot_class_{target_class}.png'))

print("Bar plots plotted and saved.")
