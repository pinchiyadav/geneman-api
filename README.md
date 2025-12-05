# GeneMAN API Setup

Generalizable single-image 3D human reconstruction using GeneMAN.

## Requirements

- NVIDIA GPU with 20GB+ VRAM
- CUDA compatible drivers
- Python 3.10+

## Quick Start

### One-Click Setup

```bash
chmod +x setup.sh
./setup.sh
```

### Usage

```bash
# Preprocess image
python preprocessing.py data/examples --output_path data/processed --recenter --enable_captioning

# Run full pipeline
sh script/run.sh

# Start API server
python api.py
```

## API Endpoints

- `GET /health` - Health check
- `POST /generate/upload` - Upload image for processing
- `GET /download/{filename}` - Download generated files

## Citation

```bibtex
@article{wang2024geneman,
  title={GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data},
  author={Wang, Wentao and others},
  journal={arXiv preprint arXiv:2411.18624},
  year={2024}
}
```
