"""
Basic PyTorch tensor manipulation examples.
This demonstrates fundamental tensor operations for deep learning.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def tensor_basics():
    # Creating tensors
    print("Creating basic tensors:")
    x = torch.tensor([1, 2, 3, 4])
    y = torch.zeros(3, 4)
    z = torch.ones(2, 3, 4)
    
    print(f"1D tensor: {x}")
    print(f"2D tensor with zeros: \n{y}")
    print(f"3D tensor with ones shape: {z.shape}")
    
    # Tensor operations
    print("\nBasic tensor operations:")
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"Matrix multiplication: {torch.matmul(a, b)}")
    
    # GPU availability
    print("\nChecking GPU availability:")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available! Creating a tensor on GPU:")
        gpu_tensor = torch.tensor([1, 2, 3], device=device)
        print(f"Tensor on device: {gpu_tensor.device}")
    else:
        print("GPU is not available, using CPU instead.")
    
    # Gradient tracking (for neural networks)
    print("\nGradient computation example:")
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"d(x^2)/dx at x=2 is: {x.grad}")

def visualize_tensor():
    # Create a 2D tensor representing a simple image
    image = torch.zeros(10, 10)
    image[2:8, 2:8] = 1.0
    image[4:6, 4:6] = 0.5
    
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='viridis')
    plt.colorbar()
    plt.title("Tensor Visualization")
    plt.savefig("tensor_visualization.png")
    print("\nTensor visualization saved as 'tensor_visualization.png'")

if __name__ == "__main__":
    print("Running basic PyTorch tensor examples:\n")
    tensor_basics()
    visualize_tensor()
    print("\nAll examples completed successfully!")
