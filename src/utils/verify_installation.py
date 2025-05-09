"""
Verification script to check if all required packages are correctly installed.
"""
import os
import sys

def check_pytorch():
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("PyTorch is not installed.")
        return False

def check_simpleitk():
    try:
        import SimpleITK as sitk
        print(f"SimpleITK version: {sitk.Version()}")
        return True
    except ImportError:
        print("SimpleITK is not installed.")
        return False

def check_monai():
    try:
        import monai
        print(f"MONAI version: {monai.__version__}")
        return True
    except ImportError:
        print("MONAI is not installed.")
        return False

def main():
    print("Checking required packages...\n")
    
    pytorch_ok = check_pytorch()
    print("")
    
    simpleitk_ok = check_simpleitk()
    print("")
    
    monai_ok = check_monai()
    print("")
    
    all_ok = pytorch_ok and simpleitk_ok and monai_ok
    
    if all_ok:
        print("All required packages are installed correctly!")
    else:
        print("Some packages are missing or not installed correctly.")
        print("Please check the output above and install missing packages.")

if __name__ == "__main__":
    main()
