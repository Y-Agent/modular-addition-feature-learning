# On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking

Code repository for the paper investigating how neural networks learn modular arithmetic through Fourier feature learning, lottery ticket hypothesis, and grokking phenomena.

## 📚 Abstract

This repository contains the implementation and experimental code for analyzing how neural networks learn modular addition tasks. We study three key phenomena:
1. **Fourier Feature Learning**: How networks decompose modular addition into sparse Fourier basis functions
2. **Lottery Ticket Hypothesis**: Identification of sparse subnetworks that achieve full performance
3. **Grokking Dynamics**: Sudden generalization after extended training on modular arithmetic tasks

## 🔧 Requirements

### Software Dependencies
- Python 3.8+
- PyTorch 1.12+
- NumPy
- Matplotlib
- Plotly
- Weights & Biases (wandb)
- Jupyter Notebook
- Seaborn
- IPyWidgets

## 📁 Project Structure

```
modular-addition-feature-learning/
├── src/                          # Main source code
│   ├── module_nn.py             # Main training script with CLI interface
│   ├── nnTrainer.py             # Training loop and optimization logic
│   ├── model_base.py            # Neural network architectures (EmbedMLP)
│   ├── mechanism_base.py        # Fourier analysis and decomposition tools
│   ├── utils.py                 # Helper functions and utilities
│   ├── configs.yaml             # Default hyperparameter configuration
│   ├── saved_models/            # Checkpoint storage directory
│   └── wandb/                   # Weights & Biases logs
├── notebooks/                    # Analysis and visualization notebooks
│   ├── empirical_insight_standard.ipynb    # Standard training analysis
│   ├── empirical_insight_grokk.ipynb       # Grokking phenomenon analysis
│   ├── lottery_mechanism.ipynb             # Lottery ticket hypothesis
│   ├── interprete_gd_dynamics.ipynb        # Gradient dynamics interpretation
│   └── decouple_dynamics_simulation.ipynb  # Frequency decoupling analysis
├── figures/                     # Generated visualizations
├── run_experiment.sh            # SLURM batch submission script
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

Modify `run_experiment.sh` for your specific cluster configuration.



