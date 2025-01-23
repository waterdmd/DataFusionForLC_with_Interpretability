#%%================================================================================================
import torch
from torch.utils.data import DataLoader
import numpy as np
from captum.attr import LayerGradCam, LayerAttribution
import segmentation_models_pytorch as smp
from Model.seg_dataset import SegmentationDatasetFusion
import os
import json

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

# Dictionary to store the results for all images
all_gradcam_analysis = {}

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

# Process each image in the dataloader
for idx, sample in enumerate(test_dataloader):
    image_test, mask = sample['image'], sample['mask']
    print(f"Processing image {idx + 1}/{len(test_dataloader)}")

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

    # Dictionary to store the results for the current image
    gradcam_analysis = {}

    # Compute and analyze GradCAM for each unique class
    for target_class in unique_classes:
        print(f"Computing GradCAM for class: {target_class}")
        # Define the GradCAM object for the current class
        lgc = LayerGradCam(lambda inputs: agg_segmentation_wrapper(inputs, int(target_class)), target_layer)
        # Compute GradCAM attributions
        gc_attr = lgc.attribute(image_test.to(DEVICE))
        upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, image_test.shape[2:])

        # Analyze the attribution values
        positive_counts = {}
        negative_counts = {}
        upsampled_gc_attr_np = upsampled_gc_attr[0, 0, :, :].detach().cpu().numpy()
        out_max_np = out_max[0, 0, :, :].cpu().numpy()

        for class_label in unique_classes:
            class_label = int(class_label)  # Ensure the class label is an integer
            positive_counts[class_label] = int(((upsampled_gc_attr_np > 0) & (out_max_np == class_label)).sum())
            negative_counts[class_label] = int(((upsampled_gc_attr_np < 0) & (out_max_np == class_label)).sum())

        gradcam_analysis[int(target_class)] = {
            'positive_counts': positive_counts,
            'negative_counts': negative_counts
        }

    # Store the analysis for the current image
    all_gradcam_analysis[f'image_{idx + 1}'] = gradcam_analysis

# Save the analysis results for all images
with open(os.path.join(output_dir, 'all_gradcam_analysis.json'), 'w') as f:
    json.dump(all_gradcam_analysis, f, indent=4)

print("GradCAM analysis for all images completed and saved.")
