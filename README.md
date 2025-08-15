# Understanding Feature Learning in Modular Addition

Code repository for investigating how neural networks learn modular arithmetic through Fourier feature decomposition.

## Abstract

This repository contains the implementation and experimental code for analyzing how neural networks learn modular addition tasks. We study the emergence of sparse Fourier features during training and characterize the learning dynamics through frequency-domain analysis.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Matplotlib
- Plotly
- Weights & Biases (wandb)
- Jupyter Notebook

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd module-addition-feature

# Install dependencies
pip install torch numpy matplotlib plotly wandb jupyter ipywidgets seaborn

# Configure Weights & Biases (optional)
wandb login
```

## Project Structure

```
module-addition-feature/
├── src/
│   ├── module_nn.py          # Main training script with CLI interface
│   ├── nnTrainer.py          # Training loop and optimization logic
│   ├── model_base.py         # Neural network architectures
│   ├── mechanism_base.py     # Fourier analysis and decomposition tools
│   ├── utils.py              # Utilities and helper functions
│   └── configs.yaml          # Default hyperparameters
├── notebooks/
│   ├── empirical_insight_standard.ipynb    # Main results analysis
│   ├── lottery_mechanism.ipynb             # Sparse network analysis
│   ├── interprete_gd_dynamics.ipynb        # Gradient flow analysis
│   └── decouple_dynamics_simulation.ipynb  # Frequency decoupling analysis
├── figures/                   # Generated figures
├── run.sh                     # SLURM submission script
└── README.md
```

## Quick Start

### Training a Model

```bash
# Train with default configuration
cd src
python module_nn.py

# Train with custom parameters
python module_nn.py --p 23 --d_mlp 512 --act_type ReLU --optimizer SGD --lr 0.1

# Run batch experiments
python module_nn.py --experiments --p 17 --num_epochs 10000
```

### Command-line Arguments

- `--p`: Prime modulus (default: 23)
- `--d_mlp`: Hidden layer width (default: 512)
- `--act_type`: Activation function (`ReLU`, `Quad`, `GeLU`)
- `--init_type`: Initialization scheme (`random`, `single-freq`)
- `--optimizer`: Optimizer choice (`SGD`, `AdamW`)
- `--lr`: Learning rate
- `--num_epochs`: Training epochs
- `--experiments`: Run multiple experimental configurations

### Running on HPC Cluster

```bash
sbatch run.sh
```

## Reproducing Results

### Main Experiments

1. **Standard Training Analysis**
   ```bash
   python src/module_nn.py --init_type random --act_type ReLU --optimizer AdamW
   ```
   Then analyze results in `notebooks/empirical_insight_standard.ipynb`

2. **Single Frequency Initialization**
   ```bash
   python src/module_nn.py --init_type single-freq --act_type Quad --optimizer SGD --lr 0.1
   ```

3. **Lottery Ticket Analysis**
   Run training, then use `notebooks/lottery_mechanism.ipynb` for sparse network analysis

### Analysis Notebooks

All notebooks in `notebooks/` can be run after training models. They load saved checkpoints from `src/saved_models/` and generate visualizations.

## Key Components

### Model Architecture
- **EmbedMLP**: Two-layer MLP with embedding layer for modular arithmetic
- Supports various activation functions (ReLU, quadratic, GeLU, etc.)
- Fourier basis decomposition for weight analysis

### Training Features
- Automatic checkpointing at regular intervals
- Weights & Biases integration for experiment tracking
- Support for different initialization schemes
- Gradient and parameter norm tracking

### Analysis Tools
- Fourier decomposition of learned weights
- Phase relationship analysis
- Sparsity measurement (Inverse Participation Ratio)
- Frequency-specific neuron tracking
