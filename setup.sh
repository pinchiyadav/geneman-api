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
    git clone --depth 1 https://github.com/3DTopia/GeneMAN.git
fi

cd GeneMAN

# Install PyTorch with CUDA
echo ""
echo "[2/8] Installing PyTorch with CUDA support..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 -q

# Install core dependencies
echo ""
echo "[3/8] Installing core dependencies..."
pip install lightning omegaconf jaxtyping typeguard controlnet_aux taming-transformers-rom1504 \
    diffusers transformers accelerate huggingface-hub numpy scipy einops \
    opencv-python imageio trimesh gradio fastapi uvicorn tqdm -q

# Install additional packages
echo ""
echo "[4/8] Installing additional packages..."
pip install "git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2" -q || true
pip install "git+https://github.com/NVlabs/nvdiffrast.git" -q
pip install pysdf PyMCubes wandb torchmetrics -q || true
pip install "git+https://github.com/ashawkey/envlight.git" -q || true
pip install "git+https://github.com/openai/CLIP.git" -q
pip install ultralytics "git+https://github.com/facebookresearch/segment-anything.git" -q

# Download models from HuggingFace
echo ""
echo "[5/8] Downloading pretrained models..."
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading GeneMAN models...')
snapshot_download('wwt117/GeneMAN', local_dir='./hf_models', repo_type='dataset')
" 2>/dev/null || echo "Model download skipped, will download on first run"

# Organize models
echo ""
echo "[6/8] Organizing model files..."
mkdir -p pretrained_models extern/tets
mv hf_models/pretrained_models/* pretrained_models/ 2>/dev/null || true
mv hf_models/tets/* extern/tets/ 2>/dev/null || true

# Download SAM model
echo ""
echo "[7/8] Downloading SAM model..."
mkdir -p pretrained_models/seg
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P pretrained_models/seg 2>/dev/null || true

# Download HumanNorm models
echo ""
echo "[8/8] Downloading HumanNorm models..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('xanderhuang/normal-adapted-sd1.5', local_dir='./pretrained_models/normal-adapted-sd1.5')
snapshot_download('xanderhuang/depth-adapted-sd1.5', local_dir='./pretrained_models/depth-adapted-sd1.5')
snapshot_download('xanderhuang/normal-aligned-sd1.5', local_dir='./pretrained_models/normal-aligned-sd1.5')
snapshot_download('xanderhuang/controlnet-normal-sd1.5', local_dir='./pretrained_models/controlnet-normal-sd1.5')
" 2>/dev/null || echo "HumanNorm models download skipped"

# Copy API files
cp ../api.py . 2>/dev/null || true
cp ../generate.py . 2>/dev/null || true

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
