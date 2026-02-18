# On the Mechanism and Dynamics of Modular Addition

### Fourier Features, Lottery Ticket, and Grokking

**Jianliang He, Leda Wang, Siyu Chen, Zhuoran Yang**
*Department of Statistics and Data Science, Yale University*

[[arXiv (coming soon)](#)] [[Blog (coming soon)](#)] [[Interactive Demo](https://huggingface.co/spaces/y-agent/modular-addition-feature-learning)]

---

## Overview

This repository provides the code for studying how a two-layer neural network learns modular arithmetic $f(x,y) = (x+y) \bmod p$. We analyze three phenomena:

1. **Fourier Feature Learning** — Each neuron independently discovers a cosine wave at a single frequency, collectively implementing a discrete Fourier transform that the network was never taught.
2. **Lottery Ticket Dynamics** — Random initialization determines which frequency each neuron will specialize in: the frequency with the best initial phase alignment wins a winner-take-all competition.
3. **Grokking** — Under partial data with weight decay, the network first memorizes, then suddenly generalizes through a three-stage process: memorization → sparsification → cleanup.

## Interactive Demo

An interactive Gradio app visualizes all results with math explanations and interactive Plotly charts:

- **9 analysis tabs** covering mechanism, dynamics, grokking, and analytical simulations
- **Interactive features**: neuron frequency inspector, logit explorer, grokking epoch slider
- **On-demand training**: generate results for any odd $p \geq 3$ directly from the app
- **Pre-computed examples** included for $p = 15, 23, 29, 31$

### Launch Locally

```bash
pip install -r requirements.txt
python hf_app/app.py
# Opens at http://localhost:7860
```

### Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) (SDK: Gradio)
2. Push the repo:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push hf main
   ```
3. The app reads from `precomputed_results/` — the included examples (p=15, 23, 29, 31) work out of the box
4. Users can generate results for additional $p$ values on-demand via the "Generate" button. New results are auto-committed back to the Space repo so they persist.

> **Tip:** For GPU-accelerated on-demand training, select a GPU runtime in your Space settings.

## Pre-computation Pipeline

The `precompute/` directory trains 5 model configurations per modulus and generates all plots + interactive JSON data. See [`precompute/README.md`](precompute/README.md) for full documentation.

### Quick Start

```bash
# Full pipeline for a single modulus (train → plots → analytical → verify)
bash precompute/run_pipeline.sh 23

# With custom d_mlp
bash precompute/run_pipeline.sh 23 --d_mlp 128

# Delete checkpoints after generating plots (saves disk space)
CLEANUP=1 bash precompute/run_pipeline.sh 23

# Batch: all odd p in [3, 99]
bash precompute/run_all.sh

# Or up to p=199
MAX_P=199 bash precompute/run_all.sh
```

### Manual Steps

```bash
# Step 1: Train all 5 configurations
python precompute/train_all.py --p 23 --output ./trained_models --resume

# Step 2: Generate model-based plots (21 PNGs + 7 JSONs)
python precompute/generate_plots.py --p 23 --input ./trained_models --output ./precomputed_results

# Step 3: Generate analytical simulation plots (2 PNGs, no model needed)
python precompute/generate_analytical.py --p 23 --output ./precomputed_results
```

### Output

Each modulus produces ~33 files in `precomputed_results/p_XXX/`:

| Category | Files | Description |
|----------|-------|-------------|
| Overview (Tab 1) | 2 PNGs + 1 JSON | Loss, IPR, phase scatter |
| Fourier Weights (Tab 2) | 3 PNGs + 1 JSON | DFT heatmaps, cosine fits, neuron spectra |
| Phase Analysis (Tab 3) | 3 PNGs | Phase distribution, alignment, magnitudes |
| Output Logits (Tab 4) | 1 PNG + 1 JSON | Logit heatmap, interactive explorer |
| Lottery Mechanism (Tab 5) | 3 PNGs | Magnitude race, phase convergence, contour |
| Grokking (Tab 6) | 5 PNGs + 3 JSONs | Loss/acc curves, memorization, weight evolution |
| Gradient Dynamics (Tab 7) | 4 PNGs | Phase alignment + DFT for Quad and ReLU |
| Decoupled Simulation (Tab 8) | 2 PNGs | Analytical ODE integration |
| Metadata | 2 JSONs | Config + training log |

> **Note:** Grokking results (Tab 6) require $p \geq 19$. Smaller values of $p$ have too few data points for a meaningful train/test split.

## The 5 Training Configurations

| Config | Activation | Optimizer | LR | Weight Decay | Data | Epochs | Used In |
|--------|-----------|-----------|-----|-------------|------|--------|---------|
| `standard` | ReLU | AdamW | 5e-5 | 0 | 100% | 5,000 | Tabs 1–4 |
| `grokking` | ReLU | AdamW | 1e-4 | 2.0 | 75% | 50,000 | Tabs 1, 6 |
| `quad_random` | Quad | AdamW | 5e-5 | 0 | 100% | 5,000 | Tab 5 |
| `quad_single_freq` | Quad | SGD | 0.1 | 0 | 100% | 5,000 | Tab 7 |
| `relu_single_freq` | ReLU | SGD | 0.01 | 0 | 100% | 5,000 | Tab 7 |

## Running a Single Experiment

For custom experiments outside the pre-computation pipeline:

```bash
cd src

# Train with default config (p=97, d_mlp=1024, ReLU, 5000 epochs)
python module_nn.py

# Train with specific parameters
python module_nn.py --p 23 --d_mlp 512 --num_epochs 5000 --lr 5e-5

# Dry run: see config without training
python module_nn.py --dry_run --p 23 --d_mlp 512
```

## Notebooks

Interactive analysis notebooks in `notebooks/`:

| Notebook | Description |
|----------|-------------|
| `empirical_insight_standard.ipynb` | Fourier weight analysis, phase distributions, output logits |
| `empirical_insight_grokk.ipynb` | Grokking stages, weight dynamics, IPR evolution |
| `lottery_mechanism.ipynb` | Neuron specialization, frequency magnitude/phase tracking |
| `interprete_gd_dynamics.ipynb` | Phase alignment under single-frequency initialization |
| `decouple_dynamics_simulation.ipynb` | Analytical gradient flow simulation |

## Setup

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for $p > 50$; CPU works for small $p$)

### Installation

```bash
git clone https://github.com/Y-Agent/modular-addition-feature-learning.git
cd modular-addition-feature-learning
pip install -r requirements.txt
```

## Project Structure

```
modular-addition-feature-learning/
├── src/                          # Core source code
│   ├── module_nn.py             # Training script with CLI
│   ├── nnTrainer.py             # Training loop and optimization
│   ├── model_base.py            # Neural network architecture (EmbedMLP)
│   ├── mechanism_base.py        # Fourier analysis and decomposition
│   ├── utils.py                 # Configuration and helpers
│   └── configs.yaml             # Default hyperparameters
├── precompute/                   # Batch training and plot generation
│   ├── run_pipeline.sh          # Full pipeline for one modulus
│   ├── run_all.sh               # Batch pipeline for all odd p
│   ├── train_all.py             # Train 5 configurations
│   ├── generate_plots.py        # Generate model-based plots + JSONs
│   ├── generate_analytical.py   # Analytical ODE simulation plots
│   └── prime_config.py          # Configurations and sizing formula
├── hf_app/                       # Gradio web application
│   └── app.py                   # Interactive visualization app
├── precomputed_results/          # Pre-computed plots and data
│   ├── p_015/                   # Results for p=15
│   ├── p_023/                   # Results for p=23
│   ├── p_029/                   # Results for p=29
│   └── p_031/                   # Results for p=31
├── notebooks/                    # Analysis and visualization notebooks
├── requirements.txt              # Python dependencies
└── README.md
```

## Citation

```bibtex
@article{he2025modular,
  title={On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking},
  author={He, Jianliang and Wang, Leda and Chen, Siyu and Yang, Zhuoran},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

[MIT License](LICENSE)
