# NECIL-HSI: Non-Exemplar Class Incremental Learning for HSI Classification

A PyTorch implementation of Dynamic Augmented Memory (DAM) based architecture for class-incremental learning on Hyperspectral Images without storing exemplars.

## Architecture Overview

```
HSI Cube → Patches → Spatial-Spectral Feature Extractor → Semantic Token (z)
                            ↓
                   Dynamic Augmented Memory (DAM)
                   ├── Refiner(z) → Q
                   ├── Extractor(z) → Km  
                   ├── Composer(z) → Vm
                   ├── SCM (Semantic Class Memory)
                   └── Differential Semantic Attention
                            ↓
                       Classifier → Predictions
```

## Installation

```bash
# Create environment
conda create -n necil_hsi python=3.10
conda activate necil_hsi

# Install PyTorch with CUDA 12.6
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
NECIL_HSI_DAM/
├── config/
│   └── config.py           # Dataset configs & hyperparameters
├── data/
│   ├── hsi_loader.py       # Load HSI .mat files
│   ├── patch_extractor.py  # Extract patches
│   └── incremental_dataset.py  # Manage incremental splits
├── models/
│   ├── feature_extractor.py    # 3D-CNN backbone
│   ├── semantic_token.py       # Token generator
│   ├── dam.py                  # Dynamic Augmented Memory
│   ├── classifier.py           # Incremental classifier
│   └── necil_model.py          # Full model
├── losses/
│   ├── cross_entropy.py        # CE loss (new classes)
│   ├── contrastive.py          # Memory alignment
│   ├── memory_stability.py     # Prevent forgetting
│   ├── calibration.py          # Classifier calibration
│   └── combined.py             # Combined NECIL loss
├── trainers/
│   └── incremental_trainer.py  # Training loop
├── utils/
│   ├── metrics.py              # Evaluation metrics
│   └── visualization.py        # Plotting
├── datasets/                   # Your HSI data (.mat files)
├── main.py                     # Entry point
└── requirements.txt
```

## Usage

### Quick Test
```bash
python main.py --dataset Indian_pines --epochs_base 5 --epochs_inc 3 --test_mode
```

### Full Training
```bash
python main.py --dataset Indian_pines --epochs_base 100 --epochs_inc 50
```

### All Options
```bash
python main.py \
    --dataset PaviaU \
    --data_dir ./datasets \
    --patch_size 11 \
    --hidden_dim 256 \
    --epochs_base 100 \
    --epochs_inc 50 \
    --batch_size 64 \
    --lr_base 0.001 \
    --lr_inc 0.0005 \
    --lambda_ce 1.0 \
    --lambda_cma 0.5 \
    --lambda_msl 0.3 \
    --lambda_cc 0.2 \
    --seed 42
```

## Supported Datasets

| Dataset | Classes | Bands | Size |
|---------|---------|-------|------|
| Indian_pines | 16 | 200 | 145×145 |
| PaviaU | 9 | 103 | 610×340 |
| PaviaC | 9 | 102 | 1096×715 |
| Salinas | 16 | 204 | 512×217 |
| Houston13 | 15 | 144 | 349×1905 |
| KSC | 13 | 176 | 512×614 |
| Botswana | 14 | 145 | 1476×256 |

## Incremental Learning Setup

- **Phase 0 (Base)**: ~50% of classes
- **Phase 1-N (Incremental)**: 2 classes per phase

## Loss Functions

| Loss | Description |
|------|-------------|
| **CE** | Cross Entropy (new class samples only) |
| **CMA** | Contrastive Memory Alignment |
| **MSL** | Memory Stability Loss |
| **CC** | Classifier Calibration |

**Total**: `L = λ₁·CE + λ₂·CMA + λ₃·MSL + λ₄·CC`

## Reference

This implementation is based on the architecture shown in the provided diagram for Non-Exemplar Class Incremental Learning with Dynamic Augmented Memory.
