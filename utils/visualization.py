import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    return plt


def generate_gradcam(model, input_tensor, target_class):
    model.eval()
    conv_output = None

    def save_output(module, input, output):
        nonlocal conv_output
        conv_output = output

    last_conv_layer = list(model.backbone.children())[-1]
    handle = last_conv_layer.register_forward_hook(save_output)

    # Forward pass
    model_output = model(input_tensor)

    # Backward pass
    model.zero_grad()
    class_output = model_output[0, target_class]
    class_output.backward()

    # Get gradients and feature maps
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight feature maps
    for i in range(conv_output.shape[1]):
        conv_output[:, i, :, :] *= pooled_gradients[i]

    # Generate heatmap
    heatmap = torch.mean(conv_output, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    heatmap /= np.max(heatmap)

    handle.remove()

    return heatmap


def plot_gradcam(image, heatmap):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title('GradCAM Heatmap')
    plt.axis('off')

    plt.tight_layout()
    return plt

