# On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking

Code repository for the paper investigating how neural networks learn modular arithmetic through Fourier feature learning, lottery ticket hypothesis, and grokking phenomena.

## Abstract

This repository contains the implementation and experimental code for analyzing how neural networks learn modular addition tasks (a + b mod p). We study three key phenomena:
1. **Fourier Feature Learning**: How networks decompose modular addition into sparse Fourier basis functions
2. **Lottery Ticket Hypothesis**: Identification of sparse subnetworks that achieve full performance
3. **Grokking Dynamics**: Sudden generalization after extended training on modular arithmetic tasks

## Setup

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for large primes; CPU works for small primes)

### Installation

```bash
git clone https://github.com/your-org/modular-addition-feature-learning.git
cd modular-addition-feature-learning
pip install -r requirements.txt
```

Weights & Biases (wandb) is optional. Install it separately if you want experiment tracking:

```bash
pip install wandb
```

## Quick Start

### Training a single model

All training is done via `src/module_nn.py`. Run from the `src/` directory:

```bash
cd src

# Train with default config (p=97, d_mlp=1024, ReLU, 5000 epochs)
python module_nn.py

# Train a specific prime with custom parameters
python module_nn.py --p 23 --d_mlp 512 --num_epochs 5000 --lr 5e-5

# Train without wandb logging
python module_nn.py --p 23 --d_mlp 512 --no_wandb

# Dry run to see the configuration without training
python module_nn.py --dry_run --p 23 --d_mlp 512
```

### Key CLI parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--p` | Prime number for modular arithmetic | 97 |
| `--d_mlp` | Number of hidden neurons | 1024 |
| `--act_type` | Activation function: `ReLU`, `GeLU`, `Quad`, `Id` | `ReLU` |
| `--init_type` | Weight initialization: `random`, `single-freq` | `random` |
| `--init_scale` | Scale factor for weight initialization | 0.1 |
| `--optimizer` | Optimizer: `AdamW`, `SGD` | `AdamW` |
| `--lr` | Learning rate | 5e-5 |
| `--weight_decay` | Weight decay for regularization | 0 |
| `--num_epochs` | Number of training epochs | 5000 |
| `--frac_train` | Fraction of data for training | 1.0 |
| `--save_models` | Save intermediate checkpoints | false |
| `--save_every` | Checkpoint frequency (epochs) | 200 |
| `--seed` | Random seed | 42 |
| `--no_wandb` | Disable wandb logging | false |

### Experiment configurations

The paper uses five training configurations per prime:

**1. Standard training** (Fourier feature learning):
```bash
python module_nn.py --p 23 --d_mlp 512 --act_type ReLU --init_type random \
    --optimizer AdamW --lr 5e-5 --num_epochs 5000 --init_scale 0.1 \
    --save_models true --save_every 200 --no_wandb
```

**2. Grokking** (delayed generalization with weight decay):
```bash
python module_nn.py --p 23 --d_mlp 512 --act_type ReLU --init_type random \
    --optimizer AdamW --lr 1e-4 --weight_decay 2 --frac_train 0.75 \
    --num_epochs 50000 --init_scale 0.1 --save_models true --save_every 200 --no_wandb
```

**3. Quadratic activation** (random init):
```bash
python module_nn.py --p 23 --d_mlp 512 --act_type Quad --init_type random \
    --optimizer AdamW --lr 5e-5 --num_epochs 5000 --init_scale 0.1 \
    --save_models true --save_every 200 --no_wandb
```

**4. Single-frequency init with Quad activation** (gradient dynamics):
```bash
python module_nn.py --p 23 --d_mlp 512 --act_type Quad --init_type single-freq \
    --optimizer SGD --lr 0.1 --num_epochs 5000 --init_scale 0.02 \
    --save_models true --save_every 200 --no_wandb
```

**5. Single-frequency init with ReLU activation** (gradient dynamics):
```bash
python module_nn.py --p 23 --d_mlp 512 --act_type ReLU --init_type single-freq \
    --optimizer SGD --lr 0.01 --num_epochs 5000 --init_scale 0.002 \
    --save_models true --save_every 200 --no_wandb
```

### Model architecture

The model is an **EmbedMLP**: one-hot embedding of two inputs (a, b), a single hidden layer with `d_mlp` neurons, and a p-class output head. The network learns to predict (a + b) mod p.

The recommended number of hidden neurons scales with p:

```
d_mlp = max(512, ceil(512/529 * p^2))
```

This maintains the ratio from the baseline experiment (p=23, d_mlp=512).

### SLURM cluster usage

For running on a SLURM cluster, modify `run_experiment.sh` with your cluster configuration and submit:

```bash
sbatch run_experiment.sh
```

### Default configuration

All defaults are in `src/configs.yaml`. CLI arguments override these values.

## Batch Pre-computation Pipeline

The `precompute/` directory contains scripts for batch training across all primes 3-199 and generating visualization plots:

```bash
# Train all 5 runs for a single prime
python precompute/train_all.py --prime 23 --output ./trained_models

# Train all primes (45 primes x 5 runs = 225 training runs)
python precompute/train_all.py --all --output ./trained_models

# Resume interrupted training (skips completed runs)
python precompute/train_all.py --all --output ./trained_models --resume

# Generate all plots from trained models
python precompute/generate_plots.py --prime 23 --input ./trained_models --output ./precomputed_results

# Generate analytical/simulation plots (no trained model needed)
python precompute/generate_analytical.py --prime 23 --output ./precomputed_results
```

## Notebooks

Interactive analysis notebooks are in `notebooks/`:

| Notebook | Description |
|----------|-------------|
| `empirical_insight_standard.ipynb` | Standard training: Fourier weight analysis, phase distributions, output logits |
| `empirical_insight_grokk.ipynb` | Grokking: memorization stages, weight dynamics, IPR evolution |
| `lottery_mechanism.ipynb` | Lottery ticket: neuron specialization, frequency magnitude/phase tracking |
| `interprete_gd_dynamics.ipynb` | Gradient dynamics: phase alignment under single-frequency initialization |
| `decouple_dynamics_simulation.ipynb` | Analytical: decoupled gradient flow simulation |
| `frequency_subset_analysis.ipynb` | Analytical: frequency/phase diversity and output quality |

## Project Structure

```
modular-addition-feature-learning/
├── src/                          # Main source code
│   ├── module_nn.py             # Training script with CLI interface
│   ├── nnTrainer.py             # Training loop and optimization logic
│   ├── model_base.py            # Neural network architecture (EmbedMLP)
│   ├── mechanism_base.py        # Fourier analysis and decomposition tools
│   ├── utils.py                 # Configuration and helper utilities
│   └── configs.yaml             # Default hyperparameter configuration
├── notebooks/                    # Analysis and visualization notebooks
├── precompute/                   # Batch training and plot generation pipeline
│   ├── prime_config.py          # Prime list, d_mlp formula, run configurations
│   ├── train_all.py             # Batch training orchestrator
│   ├── generate_plots.py        # Plot generation from trained models
│   ├── generate_analytical.py   # Analytical/simulation plot generation
│   ├── neuron_selector.py       # Automated neuron selection for plots
│   └── grokking_stage_detector.py  # Grokking stage boundary detection
├── hf_app/                       # Hugging Face Spaces application
│   ├── app.py                   # Gradio web app
│   └── requirements.txt         # HF Space dependencies (CPU PyTorch)
├── figures/                      # Generated visualizations
├── requirements.txt              # Python dependencies
├── run_experiment.sh             # SLURM batch submission script
└── README.md
```
