# DETR - Detection Transformer Implementation

Object detection implementation using Detection Transformer (DETR) architecture trained on Pascal VOC dataset.

## Project Structure

- **model.py** - DETR model architecture (Backbone, Encoder, Decoder)
- **dataset.py** - VOC dataset loader and preprocessing
- **loss.py** - Hungarian matching and loss computation
- **train.py** - Training loop
- **config.py** - Configuration parameters
- **visualization.py** - Visualization utilities
- **main.py** - Dataset initialization

## Setup

### Requirements
```bash
pip install torch torchvision tqdm scipy matplotlib numpy
```

### Dataset
The code automatically downloads VOC2007 and VOC2012 datasets. Update `BASE_DIR` in `config.py` if needed.

## Configuration

Edit `config.py` to modify:
- **dataset_config**: Dataset paths and parameters
- **model_config**: Model architecture and training hyperparameters
- **train_config**: Training settings (batch size, learning rate, epochs)

## Training

```bash
python train.py
```

## Key Features

- ResNet backbone (50 or 101)
- Transformer encoder-decoder architecture
- Hungarian matching for bipartite assignment
- Multi-scale positional encoding
- GIoU + L1 + Classification loss

## Model Details

- **Backbone**: ResNet50/101 with frozen batch norm
- **Encoder**: 4 layers, 8 attention heads
- **Decoder**: 4 layers, 8 attention heads
- **Queries**: 25 learnable object queries
- **Classes**: 21 (20 VOC classes + background)