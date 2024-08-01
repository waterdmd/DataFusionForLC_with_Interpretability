# #%%================================================================================================
import torch
from torch.utils.data import DataLoader
import numpy as np
from captum.attr import LayerGradCam, LayerAttribution
import segmentation_models_pytorch as smp
from Model.seg_dataset import SegmentationDatasetFusion
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model setup
model = smp.MAnet(
    encoder_name="mit_b5",
    encoder_weights=None,
    in_channels=21,
    classes=9,
    encoder_depth=5,
    activation='softmax'
)
model.eval()
model.to(DEVICE)
model_path = '/data/models/Fusion/MANET_MIT/SAR/MAnet_MiT_epochs_200_crossentropy_state_dict.pth'
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
print("Model loaded successfully")

# Load the dataset
test_dir = '/data/data/Fusion/Val'
test_ds = SegmentationDatasetFusion(data_path=test_dir)
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

# Create output directory for saving results
output_dir = 'gradcam_results'
os.makedirs(output_dir, exist_ok=True)

# Dictionary to map class indices to names
label_dict_10 = {
    'Alfalfa/Hay': [0],
    'Cotton': [1],
    'Pecan': [2],
    'Othercrops': [3],
    'Forest/shrublands': [4],
    'Grassland/Barren': [5],
    'Water bodies': [6],
    'Developed': [7],
    'Other': [8],
}

# Invert the dictionary to map indices to class names
index_to_class = {v[0]: k for k, v in label_dict_10.items()}

# List to store GradCAM values for each band
gradcam_values = {band: [] for band in range(21)}

# Define a wrapper function to aggregate outputs for GradCAM
def agg_segmentation_wrapper(inputs, target_class):
    model_output = model(inputs)
    target_output = model_output[:, target_class, :, :].sum().unsqueeze(0)
    return target_output

# Identify the last convolutional layer
def find_last_conv_layer(model):
    layers = list(model.named_modules())
    for name, layer in reversed(layers):
        if isinstance(layer, torch.nn.Conv2d):
            return name, layer
    raise ValueError("No Conv2d layer found in the model")

target_layer_name, target_layer = find_last_conv_layer(model)
print(f"Using target layer: {target_layer_name}")

CLASS = 7  # Change this to the desired class for analysis
class_name = index_to_class[CLASS]  # Get the class name for the plot title

# Process the first 100 images in the dataloader that contain the specified class in the ground truth
valid_image_count = 0
for idx, sample in enumerate(test_dataloader):
    if valid_image_count >= 500:
        break
    image_test, mask = sample['image'], sample['mask']
    unique_classes_gt = torch.unique(mask).cpu().numpy()

    if CLASS not in unique_classes_gt:
        continue

    valid_image_count += 1
    print(f"Processing valid image {valid_image_count}/500 (dataset index {idx + 1})")

    for band in range(21):
        # Create a copy of the image with only the current band retained
        modified_image = torch.zeros_like(image_test)
        modified_image[:, band, :, :] = image_test[:, band, :, :]

        # Compute GradCAM for the specified class
        lgc = LayerGradCam(lambda inputs: agg_segmentation_wrapper(inputs, CLASS), target_layer)
        gc_attr = lgc.attribute(modified_image.to(DEVICE))
        upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, image_test.shape[2:])
        upsampled_gc_attr_np = upsampled_gc_attr[0, 0, :, :].detach().cpu().numpy()

        # Clip the GradCAM values to the range [-1, 1]
        upsampled_gc_attr_np = np.clip(upsampled_gc_attr_np, -1, 1)

        # Store the GradCAM values for the current band
        gradcam_values[band].append(upsampled_gc_attr_np)

# Aggregate GradCAM values across all images
aggregated_gradcam_values = {band: np.concatenate(gradcam_values[band]) for band in range(21)}

# Convert numpy arrays to lists for JSON serialization
aggregated_gradcam_values_serializable = {band: values.tolist() for band, values in aggregated_gradcam_values.items()}

# Save the analysis results
with open(os.path.join(output_dir, f'gradcam_values_{CLASS}.json'), 'w') as f:
    json.dump(aggregated_gradcam_values_serializable, f, indent=4)

# Create a DataFrame for plotting
data = []
for band, values in aggregated_gradcam_values.items():
    for value in values.flatten():
        data.append((f'Band {band+1}', value))
df = pd.DataFrame(data, columns=['Band', 'GradCAM Value'])

# Plot the data
sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(12, 8))
sns.despine(bottom=True, left=True)

# Show each observation with a scatterplot, with matching colors
palette = sns.color_palette("viridis", n_colors=len(df['Band'].unique()))
sns.violinplot(
    data=df, x="GradCAM Value", y="Band", palette=palette, inner=None, scale='width', alpha=.6, zorder=1, legend=False,
)

# Show the conditional means
sns.pointplot(
    data=df, x="GradCAM Value", y="Band", palette=["black"] * len(palette), errorbar=None,
    markers="d", markersize=4, linestyle="none",
)

# Improve the legend and customize the plot
ax.set_yticklabels([f'Band {i+1}' for i in range(21)], fontsize=14)
ax.set_ylabel('', fontsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title(f'Impact of Each Band on GradCAM of Class {CLASS}: ({class_name})', fontsize=16)
plt.xlabel('GradCAM Value', fontsize=14)
plt.savefig(os.path.join(output_dir, f'impact_of_each_band_{CLASS}.png'), bbox_inches='tight', pad_inches=0)
plt.close()

print("GradCAM analysis for all bands completed and saved.")

