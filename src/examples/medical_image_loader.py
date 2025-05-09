"""
Example medical image loading using SimpleITK and conversion to PyTorch tensors.
"""
import os
import SimpleITK as sitk
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_sample_image():
    """
    This function would typically load an actual medical image from a dataset.
    Since we don't have a dataset yet, we'll create a synthetic vessel-like image.
    """
    # Create a synthetic vessel image (a tube-like structure)
    size = [100, 100, 100]
    image = sitk.Image(size, sitk.sitkFloat32)
    image.SetSpacing([1.0, 1.0, 1.0])
    
    # Fill the image with a synthetic vessel
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                # Create a tube along the z-axis
                dist_from_center = np.sqrt((x - size[0]//2)**2 + (y - size[1]//2)**2)
                if 15 <= dist_from_center <= 20:  # Vessel wall
                    image[x, y, z] = 1.0
    
    print("Created a synthetic vessel image")
    return image

def sitk_to_torch(sitk_image):
    """Convert SimpleITK image to PyTorch tensor."""
    # Get numpy array from SimpleITK image
    array = sitk.GetArrayFromImage(sitk_image)
    
    # Convert to PyTorch tensor (add batch dimension)
    tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)
    print(f"Converted to PyTorch tensor with shape: {tensor.shape}")
    return tensor

def visualize_slice(tensor, slice_idx=50):
    """Visualize a slice from the 3D tensor."""
    # Extract a single slice
    slice_data = tensor[0, 0, slice_idx].numpy()
    
    plt.figure(figsize=(5, 5))
    plt.imshow(slice_data, cmap='gray')
    plt.colorbar()
    plt.title(f"Vessel Cross-Section (Slice {slice_idx})")
    plt.savefig("vessel_slice.png")
    print("Vessel slice visualization saved as 'vessel_slice.png'")

def apply_simple_processing(tensor):
    """Apply a simple processing step (Gaussian smoothing in this case)."""
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)
    
    # Apply a simple blur using convolution
    kernel_size = 5
    kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size) / (kernel_size**3)
    kernel = kernel.to(device)
    
    # Need to pad the tensor for convolution
    padded = torch.nn.functional.pad(tensor, (kernel_size//2,)*6, mode='reflect')
    
    # Apply 3D convolution
    smoothed = torch.nn.functional.conv3d(padded, kernel)
    print(f"Applied Gaussian smoothing, output shape: {smoothed.shape}")
    
    return smoothed.cpu()

def main():
    print("Loading and processing a synthetic medical vessel image:\n")
    
    # Load or create a sample medical image
    sitk_image = load_sample_image()
    
    # Convert to PyTorch tensor
    tensor = sitk_to_torch(sitk_image)
    
    # Visualize a slice
    visualize_slice(tensor)
    
    # Apply some processing
    processed_tensor = apply_simple_processing(tensor)
    
    # Visualize processed result
    visualize_slice(processed_tensor, slice_idx=50)
    
    print("\nMedical image loading and processing example completed successfully!")

if __name__ == "__main__":
    main()
