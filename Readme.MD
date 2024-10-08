# Multi-label Chest X-ray Classification with Self-Supervised Learning

## Project Overview

This project implements a multi-label classification system for chest X-rays, focusing on detecting multiple thoracic diseases simultaneously. It utilizes self-supervised learning techniques, specifically SimCLR (Simple Framework for Contrastive Learning of Visual Representations), to improve feature extraction from unlabeled data. The system also incorporates a novel correlation learning module to capture dependencies between different diseases.

## Key Features

- Self-supervised pretraining using SimCLR
- Multi-label classification with binary classifiers for each disease
- Correlation learning module to capture disease interdependencies
- Adaptive Class Balance Augmentation (ACBA) for handling class imbalance
- Focal Weighted Cross-Entropy Loss for imbalanced learning
- Distributed training support for multi-GPU setups
- Comprehensive ablation study framework

```bash
## Project Structure
project/
│
├── main.py                 # Main entry point for running experiments
├── train.py                # Training functions for SimCLR and classifiers
├── ablation_study.py       # Code for running ablation studies
├── evaluate.py             # Evaluation functions
│
├── models/
│   ├── efficientnet.py     # EfficientNet backbone with attention
│   ├── simclr.py           # SimCLR model and loss
│   ├── binary_classifiers.py # Ensemble of binary classifiers
│   ├── correlation_learning.py # Correlation learning module
│   └── meta_learner.py     # Meta-learner for refining predictions
│
├── data/
│   ├── dataset.py          # Dataset and DataLoader definitions
│   └── augmentations.py    # Data augmentation, ACBA
│
├── utils/
│   ├── loss.py             # Custom FWCE loss functions
│   ├── metrics.py          # Evaluation metrics
│   └── training_utils.py   # Utility functions for training
│
└── config/
└── config.yaml         # Configuration file for experiments
```

## Installation

1. Clone this repository:

git clone https://github.com/ghazalehmirzaee/ChestXray14-MultiLabel-Classification.git
cd chest-xray-classification

2. Install the required packages:
pip install -r requirements.txt

3. Set up the dataset in the appropriate directory and update the `config.yaml` file with the correct paths.

## Usage

### Training

To run the full training pipeline (SimCLR pretraining + classifier training):

python main.py --config config/config.yaml --mode train

### Evaluation

To evaluate a trained model:
python main.py --config config/config.yaml --mode evaluate

### Ablation Study

To run the ablation study comparing models with and without correlation learning:
python main.py --config config/config.yaml --mode ablation

## Configuration

Adjust the `config/config.yaml` file to modify hyperparameters, data paths, and other settings. Key configurations include:

- `model`: Model architecture settings
- `data`: Dataset paths and settings
- `training`: Training hyperparameters
- `wandb`: Weights & Biases logging settings

## Results

The project includes comprehensive logging and visualization using Weights & Biases. Key metrics tracked include:

- Training and validation loss
- AUC-ROC scores
- Mean Average Precision
- F1 scores
- Learning rate schedules

Detailed ablation study results comparing models with and without correlation learning are also provided.

## Acknowledgments

- The ChestX-ray14 dataset provided by the NIH Clinical Center
- The SimCLR paper by Chen et al.
- The EfficientNet implementation by lukemelas

