# JamMa-SLAM Integration Environment
# Base: nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# Python: 3.9 (via Miniconda)
# Date: 2026-03-25

# 1. causal-conv1d (pre-built wheel)
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.0/causal_conv1d-1.1.0+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

# 2. mamba-ssm (pre-built wheel)
pip install https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

# 3. torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# ── Core: PyTorch (CUDA 11.8) ──────────────────────────────────────────
torch==2.0.1+cu118
torchvision==0.15.2+cu118
torchaudio==2.0.2+cu118
--extra-index-url https://download.pytorch.org/whl/cu118

# ── Mamba (pre-built wheels, cxx11abi=FALSE, cu118, py39) ──────────────
# Install via:
# pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.0/causal_conv1d-1.1.0+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
# pip install https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
causal-conv1d==1.1.0
mamba-ssm==1.1.1

# ── torch-scatter ───────────────────────────────────────────────────────
# Install via:
# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
torch-scatter==2.1.2+pt20cu118

# ── SLAM & Geometry ─────────────────────────────────────────────────────
lietorch==0.3
droid_backends==0.0.0

# ── JamMa Dependencies ──────────────────────────────────────────────────
pytorch-lightning==1.9.5
lightning-utilities==0.15.2
einops==0.8.1
kornia==0.7.0
yacs==0.1.8
h5py==3.11.0
poselib==2.0.4
timm==1.0.15
albumentations==0.5.1
scikit-image==0.21.0
loguru==0.7.3
transformers==4.35.2
tokenizers==0.15.2
safetensors==0.7.0

# ── DROID-SLAM Dependencies ─────────────────────────────────────────────
evo==1.31.1
open3d==0.18.0
gdown==5.2.1
tensorboard==2.20.0
tensorboard-data-server==0.7.2
opencv-python==4.8.1.78
scipy==1.10.1
numpy==1.26.4
PyYAML==6.0.3
tqdm==4.67.1

# ── open3d sub-dependencies ─────────────────────────────────────────────
addict==2.4.0
ConfigArgParse==1.7.5
ipywidgets==8.1.8
nbformat==5.10.4
pandas==2.3.3
pyquaternion==0.9.9
scikit-learn==1.6.1
dash==4.1.0
plotly==6.6.0

# ── evo sub-dependencies ────────────────────────────────────────────────
argcomplete==3.6.3
colorama==0.4.6
natsort==8.4.0
numexpr==2.10.2
rosbags==0.9.23
seaborn==0.13.2

# ── General ─────────────────────────────────────────────────────────────
triton==2.0.0
ninja==1.13.0
cmake==3.25.0
matplotlib==3.9.4
pillow==11.3.0
requests==2.32.5
packaging==25.0
pyDeprecate==0.3.2
torchmetrics==0.7.0
thop==0.1.1.post2209072238
shapely==2.0.7
imgaug==0.4.0
bs4==0.0.2
sympy==1.14.0
networkx==3.2.1
filelock==3.19.1
fsspec==2025.10.0
huggingface_hub==0.36.2
