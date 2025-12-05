#!/bin/bash
# GeneMAN One-Click Setup Script
# Generalizable single-image 3D human reconstruction

set -e

echo "=============================================="
echo "GeneMAN One-Click Setup"
echo "=============================================="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA GPU required (20GB+ VRAM)."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Clone the official repository
echo ""
echo "[1/8] Cloning GeneMAN repository..."
if [ -d "GeneMAN" ]; then
    echo "Repository already exists, updating..."
    cd GeneMAN && git pull && cd ..
else
    git clone --depth 1 --progress https://github.com/3DTopia/GeneMAN.git
fi

cd GeneMAN

# Install PyTorch with CUDA
echo ""
echo "[2/8] Installing PyTorch with CUDA support..."
pip install --progress-bar on torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
echo ""
echo "[3/8] Installing core dependencies..."
pip install --progress-bar on lightning omegaconf jaxtyping typeguard controlnet_aux taming-transformers-rom1504 \
    diffusers transformers accelerate huggingface-hub numpy==1.26.4 scipy einops \
    opencv-python imageio trimesh gradio fastapi uvicorn tqdm

# Install additional packages
echo ""
echo "[4/8] Installing additional packages..."
echo "Installing nerfacc..."
pip install --progress-bar on "git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2" || echo "nerfacc skipped"
echo "Installing nvdiffrast..."
pip install --progress-bar on "git+https://github.com/NVlabs/nvdiffrast.git"
echo "Installing other packages..."
pip install --progress-bar on pysdf PyMCubes wandb torchmetrics || echo "Some packages skipped"
pip install --progress-bar on "git+https://github.com/ashawkey/envlight.git" || echo "envlight skipped"
pip install --progress-bar on "git+https://github.com/openai/CLIP.git"
pip install --progress-bar on ultralytics "git+https://github.com/facebookresearch/segment-anything.git"

# Download models from HuggingFace
echo ""
echo "[5/8] Downloading pretrained models..."
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading GeneMAN models from HuggingFace...')
snapshot_download('wwt117/GeneMAN', local_dir='./hf_models', repo_type='dataset')
print('Download complete!')
" || echo "Model download skipped, will download on first run"

# Organize models
echo ""
echo "[6/8] Organizing model files..."
mkdir -p pretrained_models extern/tets
mv hf_models/pretrained_models/* pretrained_models/ 2>/dev/null || true
mv hf_models/tets/* extern/tets/ 2>/dev/null || true
echo "Models organized."

# Download SAM model
echo ""
echo "[7/8] Downloading SAM model..."
mkdir -p pretrained_models/seg
wget --progress=bar:force https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P pretrained_models/seg 2>&1 || echo "SAM download skipped"

# Download HumanNorm models
echo ""
echo "[8/8] Downloading HumanNorm models..."
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading normal-adapted-sd1.5...')
snapshot_download('xanderhuang/normal-adapted-sd1.5', local_dir='./pretrained_models/normal-adapted-sd1.5')
print('Downloading depth-adapted-sd1.5...')
snapshot_download('xanderhuang/depth-adapted-sd1.5', local_dir='./pretrained_models/depth-adapted-sd1.5')
print('Downloading normal-aligned-sd1.5...')
snapshot_download('xanderhuang/normal-aligned-sd1.5', local_dir='./pretrained_models/normal-aligned-sd1.5')
print('Downloading controlnet-normal-sd1.5...')
snapshot_download('xanderhuang/controlnet-normal-sd1.5', local_dir='./pretrained_models/controlnet-normal-sd1.5')
print('All models downloaded!')
" || echo "HumanNorm models download skipped"

# Copy API files
echo ""
echo "Copying API files..."
cd ..
cp api.py GeneMAN/ 2>/dev/null || true
cd GeneMAN

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Usage:"
echo "  # Preprocess image:"
echo "  python preprocessing.py data/examples --output_path data/processed --recenter --enable_captioning"
echo ""
echo "  # Generate 3D human:"
echo "  sh script/run.sh"
echo ""
echo "  # Start API server:"
echo "  python api.py"
echo ""
