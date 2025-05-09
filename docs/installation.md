# Development Environment Setup

## Prerequisites
- Python 3.8+ installed
- CUDA-capable GPU recommended (for faster training)
- Git installed

## Step 1: Clone the Repository
```bash
git clone https://github.com/dayaiit/AI-in-Medical-Imaging.git
cd AI-in-Medical-Imaging
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
# Install PyTorch with CUDA support (adjust based on your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install SimpleITK
pip install SimpleITK

# Install MONAI
pip install monai

# Install other dependencies
pip install numpy matplotlib jupyter pandas scikit-learn
python src/utils/verify_installation.py
