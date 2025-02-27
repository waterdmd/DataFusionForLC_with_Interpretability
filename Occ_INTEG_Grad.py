import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from captum.attr import LayerGradCam, LayerAttribution, visualization as viz
from captum.attr import IntegratedGradients, Occlusion
import segmentation_models_pytorch as smp
from Model.seg_dataset import SegmentationDatasetFusion
import os
from skimage import exposure

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

# Define a forward function for Integrated Gradients and Occlusion
def ig_occlusion_forward_func(inputs):
    model_output = model(inputs)
    # Return the mean value of the probabilities for all classes
    return model_output.mean(dim=[2, 3])

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

# Create output directory for saving plots
output_dir = 'gradcam_plots'
os.makedirs(output_dir, exist_ok=True)

# Define a colormap and labels for the segmentation output
cmap = ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'orange'])
labels = ['Class 0: Alfalfa/\nHay', 'Class 1: Cotton ', 'Class 2: Pecan', 'Class 3: Other\nCrops', 'Class 4: Forrest/\nShrubland', 'Class 5: Grassland/\n Barren', 'Class 6: Water\nbodies', 'Class 7: Developed', 'Class 8: Background']

# Visualize the original image and save the plot
# Function to apply gamma correction using skimage
def adjust_gamma(image, gamma=1.0):
    return exposure.adjust_gamma(image, gamma)

# Assuming 'image_test' is your tensor
image_np = np.transpose(image_test[0, 0:3, :, :].cpu().numpy(), (1, 2, 0))

# Adjust gamma to increase brightness
gamma_value = 0.5  # Less than 1 to increase brightness
bright_image = adjust_gamma(image_np, gamma=gamma_value)

# Plot and save the original image
plt.imshow(image_np)
plt.axis('off')
plt.title('Original Image')
plt.savefig(os.path.join(output_dir, 'original_image.png'), bbox_inches='tight', pad_inches=0)
plt.close()

# Plot and save the brightened image
plt.imshow(bright_image)
plt.axis('off')
plt.title('Brightened Image')
plt.savefig(os.path.join(output_dir, 'Input_image.png'), bbox_inches='tight', pad_inches=0)
plt.close()

# Visualize the ground truth segmentation and save the plot
mask_np = mask[0].cpu().numpy().squeeze()
plt.imshow(mask_np, cmap=cmap, vmin=0, vmax=8)  # Using the same colormap to distinguish classes
cbar = plt.colorbar(ticks=np.arange(len(labels)))
cbar.ax.set_yticklabels(labels)
plt.axis('off')
plt.title('Ground Truth Segmentation')
plt.savefig(os.path.join(output_dir, 'ground_truth_CDL.png'), bbox_inches='tight', pad_inches=0)
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

# Define color maps for visualizations
integrated_gradients_cmap = LinearSegmentedColormap.from_list('white_to_dark', ['white', 'black'])
occlusion_cmap = LinearSegmentedColormap.from_list('white_to_dark_green', ['white', 'darkgreen'])

# Custom normalization function
def normalize_attr(attr):
    min_val = torch.min(attr)
    max_val = torch.max(attr)
    if max_val - min_val != 0:
        attr = (attr - min_val) / (max_val - min_val)
    else:
        attr = attr - min_val
    return attr

# Compute and visualize GradCAM, Integrated Gradients, and Occlusion for each unique class
for target_class in unique_classes:
    print(f"Visualizing GradCAM, Integrated Gradients, and Occlusion for class: {target_class}")
    
    # GradCAM
    lgc = LayerGradCam(lambda inputs: agg_segmentation_wrapper(inputs, target_class), target_layer)
    gc_attr = lgc.attribute(image_test.to(DEVICE))
    upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, image_test.shape[2:])
    
    print('GradCAM attribution shape:', upsampled_gc_attr.shape)
    
    # Integrated Gradients
    ig = IntegratedGradients(ig_occlusion_forward_func)
    ig_attr = ig.attribute(image_test.to(DEVICE), target=int(target_class), n_steps=200)
    
    # Occlusion
    occlusion = Occlusion(ig_occlusion_forward_func)
    occlusion_attr = occlusion.attribute(image_test.to(DEVICE), strides=(3, 8, 8), target=int(target_class), sliding_window_shapes=(3, 15, 15), baselines=0)
    
    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    try:
        # GradCAM Visualization
        viz.visualize_image_attr(upsampled_gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                 original_image=image_np,
                                 sign="all",
                                 method="blended_heat_map",
                                 plt_fig_axis=(fig, axes[0]),
                                 show_colorbar=True,
                                 title=f'GradCAM for Class {target_class}')
        
        # Integrated Gradients Visualization
        viz.visualize_image_attr(ig_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                 original_image=image_np,
                                 sign="positive",
                                 method="heat_map",
                                 cmap=integrated_gradients_cmap,
                                 plt_fig_axis=(fig, axes[1]),
                                 show_colorbar=True,
                                 title=f'Integrated Gradients for Class {target_class}')
        
        # Occlusion Visualization
        viz.visualize_image_attr(occlusion_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                 original_image=image_np,
                                 sign="positive",
                                 method="heat_map",
                                 cmap=occlusion_cmap,
                                 plt_fig_axis=(fig, axes[2]),
                                 show_colorbar=True,
                                 title=f'Occlusion for Class {target_class}')

    except AssertionError as e:
        if "scale factor = 0" in str(e):
            print(f"Normalization error for class {target_class}, applying custom normalization.")
            occlusion_attr = normalize_attr(occlusion_attr)
            viz.visualize_image_attr(occlusion_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                     original_image=image_np,
                                     sign="positive",
                                     method="heat_map",
                                     cmap=occlusion_cmap,
                                     plt_fig_axis=(fig, axes[2]),
                                     show_colorbar=True,
                                     title=f'Occlusion for Class {target_class} (normalized)')
        else:
            raise e
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'interpretability_class_{target_class}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
