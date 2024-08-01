#  !/usr/bin/env python
# # coding: utf-8

# # # Semantic Segmentation with Captum



# #%%
# from torch.utils.data import DataLoader
# from PIL import Image
# import matplotlib.pyplot as plt
# import torch
# import numpy as np
# import segmentation_models_pytorch as smp
# from torchvision import models
# from torchvision import transforms
# from Model.seg_dataset import SegmentationDatasetTwoMonths, SegmentationDataset, SegmentationDatasetSeasonal,SegmentationDatasetRGB,SegmentationDatasetFusion
# from captum.attr import visualization as viz
# from captum.attr import LayerGradCam, FeatureAblation, LayerActivation, LayerAttribution
# #%%
# # Default device
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEVICE)

# # We can now load the pre-trained segmentation model from torchvision, which is trained on a subset of COCO Train 2017 and define input preprocessing transforms.

# # In[2]:

# #%% Model setup
# model = smp.MAnet(
#     encoder_name="mit_b5",
#     encoder_weights=None,
#     in_channels=21,
#     classes=9,
#     encoder_depth=5,
#     activation='softmax'
# )
# model.eval()
# model.to(DEVICE)
# model_path = '/data/models/Fusion/MANET_MIT/SAR/MAnet_MiT_epochs_200_crossentropy_state_dict.pth'
# #%% Load weights
# EPOCHS = 200

# model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
# print("model is loaded seccessfully")



# #%%
# #load the dataset
# test_dir = '/data/data/Fusion/Val'
# # test_ds = SegmentationDatasetTwoMonths(data_path=test_dir)
# test_ds = SegmentationDatasetFusion(data_path=test_dir)
# test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)  
# #%%


# #%% Pick a test image and show it
# Sample = next(iter(test_dataloader))
# image_test, mask = Sample['image'], Sample['mask']
# # Convert the tensor to a numpy array and transpose it to the correct shape
# image_np = np.transpose(image_test[0, 0:3, :, :].cpu().numpy(), (1, 2, 0))

# # Plot the image
# plt.imshow(image_np)
# plt.axis('off')  # Remove the axis
# plt.savefig('original_image.png', bbox_inches='tight', pad_inches=0)  # Save the image
# plt.show()  # Display the image

# out = model(image_test.to(DEVICE).float())

# print(f'model output shape: {out.shape}')
# out_max = torch.argmax(out, dim=1, keepdim=False)
# print(f'out_max shape: {out_max.shape}')

# out_image = out_max[0].cpu().numpy()    
# plt.imshow(out_image)
# plt.axis('off')  # Remove the axis
# plt.savefig('segmentation_output.png', bbox_inches='tight', pad_inches=0)  # Save the image


# # # """
# # def agg_segmentation_wrapper(inp):
# #     model_out = model(inp)
# #     # Creates binary matrix with 1 for original argmax class for each pixel
# #     # and 0 otherwise. Note that this may change when the input is ablated
# #     # so we use the original argmax predicted above, out_max.
# #     selected_inds = torch.zeros_like(model_out[0:1]).scatter_(1, out_max, 1)
# #     return (model_out * selected_inds).sum(dim=(2,3))


# # # # Alternate wrapper, simply summing each output channel
# # def wrapper(inp):
# #   return model(inp)['out'].sum(dim=(2,3))



# # In[19]:
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from captum.attr import LayerGradCam, LayerAttribution, visualization as viz
import segmentation_models_pytorch as smp
from Model.seg_dataset import SegmentationDatasetFusion
import os

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

# Pick a test image
sample = next(iter(test_dataloader))
image_test, mask = sample['image'], sample['mask']
print(f"Test image shape: {image_test.shape}")

# Define a wrapper function to aggregate outputs for GradCAM
def agg_segmentation_wrapper(inputs, target_class):
    model_output = model(inputs)
    # Sum the outputs over the spatial dimensions to get a scalar value
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

# Perform forward pass to get model output
with torch.no_grad():
    model_output = model(image_test.to(DEVICE).float())

# Get the class index with the highest score for each pixel
out_max = torch.argmax(model_output, dim=1, keepdim=True)
print(f"Model output shape: {model_output.shape}")
print(f"Out_max shape: {out_max.shape}")

# Identify unique classes in the ground truth and predicted output
unique_classes_gt = torch.unique(mask).cpu().numpy()
unique_classes_pred = torch.unique(out_max).cpu().numpy()
unique_classes = np.union1d(unique_classes_gt, unique_classes_pred)
print(f"Unique classes in ground truth and prediction: {unique_classes}")

#%%================================================================================================

# Create output directory for saving plots
output_dir = 'gradcam_plots'
os.makedirs(output_dir, exist_ok=True)

# Define a colormap and labels for the segmentation output
cmap = ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'orange'])
labels = ['Class 0: Alfalfa/\nHay', 'Class 1: Cotton ', 'Class 2: Pecan', 'Class 3: Other\nCrops', 'Class 4: Forrest/\nShrubland', 'Class 5: Grassland/\n Barren', 'Class 6: Water\nbodies', 'Class 7: Developed', 'Class 8: Background']

# Visualize the original image and save the plot
image_np = np.transpose(image_test[0, 0:3, :, :].cpu().numpy(), (1, 2, 0))
plt.imshow(image_np)
plt.axis('off')
plt.title('Original Image')
plt.savefig(os.path.join(output_dir, 'original_image.png'), bbox_inches='tight', pad_inches=0)
plt.close()

# Visualize the ground truth segmentation and save the plot
mask_np = mask[0].cpu().numpy().squeeze()
plt.imshow(mask_np, cmap=cmap, vmin=0, vmax=8)  # Using the same colormap to distinguish classes
cbar = plt.colorbar(ticks=np.arange(len(labels)))
cbar.ax.set_yticklabels(labels)
plt.axis('off')
plt.title('Ground Truth Segmentation')
plt.savefig(os.path.join(output_dir, 'ground_truth_segmentation.png'), bbox_inches='tight', pad_inches=0)
plt.close()

# Visualize the predicted segmentation output with color bar and save the plot
out_image = out_max[0].cpu().numpy().squeeze()
plt.imshow(out_image, cmap=cmap, vmin=0, vmax=8)  # Using the same colormap to distinguish classes
cbar = plt.colorbar(ticks=np.arange(len(labels)))
cbar.ax.set_yticklabels(labels)
plt.axis('off')
plt.title('Predicted Segmentation Output')
plt.savefig(os.path.join(output_dir, 'segmentation_output.png'), bbox_inches='tight', pad_inches=0)
plt.close()

# Compute and visualize GradCAM for each unique class
for target_class in unique_classes:
    print(f"Visualizing GradCAM for class: {target_class}")
    # Define the GradCAM object for the current class
    lgc = LayerGradCam(lambda inputs: agg_segmentation_wrapper(inputs, target_class), target_layer)
    # Compute GradCAM attributions
    gc_attr = lgc.attribute(image_test.to(DEVICE))
    upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, image_test.shape[2:])
    
    # Visualize and save the results
    fig, ax = plt.subplots(figsize=(10, 10))
    viz.visualize_image_attr(upsampled_gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                             original_image=image_np,
                             sign="all",
                             method="blended_heat_map",
                             plt_fig_axis=(fig, ax),
                             show_colorbar=True,
                             title=f'GradCAM for Class {target_class}')
    plt.savefig(os.path.join(output_dir, f'gradcam_class_{target_class}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()


#%%================================================================================================
# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# from captum.attr import LayerGradCam, LayerAttribution
# import segmentation_models_pytorch as smp
# from Model.seg_dataset import SegmentationDatasetFusion
# import os
# import json

# # Define the device
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {DEVICE}")

# # Model setup
# model = smp.MAnet(
#     encoder_name="mit_b5",
#     encoder_weights=None,
#     in_channels=21,
#     classes=9,
#     encoder_depth=5,
#     activation='softmax'
# )
# model.eval()
# model.to(DEVICE)
# model_path = '/data/models/Fusion/MANET_MIT/SAR/MAnet_MiT_epochs_200_crossentropy_state_dict.pth'
# model.load_state_dict(torch.load(model_path, map_location=DEVICE))
# print("Model loaded successfully")

# # Load the dataset
# test_dir = '/data/data/Fusion/Val'
# test_ds = SegmentationDatasetFusion(data_path=test_dir)
# test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

# # Create output directory for saving results
# output_dir = 'gradcam_results'
# os.makedirs(output_dir, exist_ok=True)

# # Dictionary to store the results for all images
# all_gradcam_analysis = {}

# # Define a wrapper function to aggregate outputs for GradCAM
# def agg_segmentation_wrapper(inputs, target_class):
#     model_output = model(inputs)
#     # Sum the outputs over the spatial dimensions to get a scalar value
#     target_output = model_output[:, target_class, :, :].sum().unsqueeze(0)
#     return target_output

# # Identify the last convolutional layer
# def find_last_conv_layer(model):
#     layers = list(model.named_modules())
#     for name, layer in reversed(layers):
#         if isinstance(layer, torch.nn.Conv2d):
#             return name, layer
#     raise ValueError("No Conv2d layer found in the model")

# target_layer_name, target_layer = find_last_conv_layer(model)
# print(f"Using target layer: {target_layer_name}")

# # Process each image in the dataloader
# for idx, sample in enumerate(test_dataloader):
#     image_test, mask = sample['image'], sample['mask']
#     print(f"Processing image {idx + 1}/{len(test_dataloader)}")

#     # Perform forward pass to get model output
#     with torch.no_grad():
#         model_output = model(image_test.to(DEVICE).float())

#     # Get the class index with the highest score for each pixel
#     out_max = torch.argmax(model_output, dim=1, keepdim=True)
#     print(f"Model output shape: {model_output.shape}")
#     print(f"Out_max shape: {out_max.shape}")

#     # Identify unique classes in the ground truth and predicted output
#     unique_classes_gt = torch.unique(mask).cpu().numpy()
#     unique_classes_pred = torch.unique(out_max).cpu().numpy()
#     unique_classes = np.union1d(unique_classes_gt, unique_classes_pred)
#     print(f"Unique classes in ground truth and prediction: {unique_classes}")

#     # Dictionary to store the results for the current image
#     gradcam_analysis = {}

#     # Compute and analyze GradCAM for each unique class
#     for target_class in unique_classes:
#         print(f"Computing GradCAM for class: {target_class}")
#         # Define the GradCAM object for the current class
#         lgc = LayerGradCam(lambda inputs: agg_segmentation_wrapper(inputs, int(target_class)), target_layer)
#         # Compute GradCAM attributions
#         gc_attr = lgc.attribute(image_test.to(DEVICE))
#         upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, image_test.shape[2:])

#         # Analyze the attribution values
#         positive_counts = {}
#         negative_counts = {}
#         upsampled_gc_attr_np = upsampled_gc_attr[0, 0, :, :].detach().cpu().numpy()
#         out_max_np = out_max[0, 0, :, :].cpu().numpy()

#         for class_label in unique_classes:
#             class_label = int(class_label)  # Ensure the class label is an integer
#             positive_counts[class_label] = int(((upsampled_gc_attr_np > 0) & (out_max_np == class_label)).sum())
#             negative_counts[class_label] = int(((upsampled_gc_attr_np < 0) & (out_max_np == class_label)).sum())

#         gradcam_analysis[int(target_class)] = {
#             'positive_counts': positive_counts,
#             'negative_counts': negative_counts
#         }

#     # Store the analysis for the current image
#     all_gradcam_analysis[f'image_{idx + 1}'] = gradcam_analysis

# # Save the analysis results for all images
# with open(os.path.join(output_dir, 'all_gradcam_analysis.json'), 'w') as f:
#     json.dump(all_gradcam_analysis, f, indent=4)

# print("GradCAM analysis for all images completed and saved.")

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
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)

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

CLASS = 6  # Change this to the desired class for analysis
class_name = index_to_class[CLASS]  # Get the class name for the plot title

# Process the first 100 images in the dataloader that contain the specified class in the ground truth
valid_image_count = 0
for idx, sample in enumerate(test_dataloader):
    if valid_image_count >= 501:
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
ax.set_xlim(-1, 1)  # Set the limits of the x-axis here
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title(f'Impact of Each Band on GradCAM of Class {CLASS}: ({class_name})', fontsize=16)
plt.xlabel('GradCAM Value', fontsize=14)
plt.savefig(os.path.join(output_dir, f'impact_of_each_band_{CLASS}.png'), bbox_inches='tight', pad_inches=0)
plt.close()

print("GradCAM analysis for all bands completed and saved.")


