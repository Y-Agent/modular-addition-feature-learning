#!/usr/bin/env python3
"""
Main plot generation script for the HF app.
Creates all model-dependent plots (Tabs 1-7) from trained checkpoints.

Usage:
    python generate_plots.py --all               # Generate for all primes
    python generate_plots.py --p 23           # Generate for a specific p
    python generate_plots.py --p 23 --input ./trained_models --output ./hf_app/precomputed_results
"""
import matplotlib
matplotlib.use('Agg')

import argparse
import json
import math
import os
import sys
import traceback

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from src.mechanism_base import (
    get_fourier_basis,
    decode_weights,
    compute_neuron,
    decode_scales_phis,
    normalize_to_pi,
)
from src.model_base import EmbedMLP
from src.utils import cross_entropy_high_precision, acc_rate
from precompute.neuron_selector import (
    select_top_neurons_by_frequency,
    select_lineplot_neurons,
    select_phase_frequency,
    select_lottery_neuron,
)
from precompute.grokking_stage_detector import detect_grokking_stages
from precompute.prime_config import compute_d_mlp, TRAINING_RUNS, MIN_P_GROKKING

# ---------- Lightweight train/test data regeneration ----------

def _gen_train_test(p, frac_train=0.75, seed=42):
    """
    Regenerate train/test split deterministically without needing a Config object.
    Mirrors the logic in utils.gen_train_test for the 'add' function.
    Returns (train_data, test_data) where each is a tensor of shape (N, 2).
    """
    import random as _random
    all_pairs = []
    for i in range(p):
        for j in range(p):
            all_pairs.append((i, j))
    data_tensor = torch.tensor(all_pairs, dtype=torch.long)
    _random.seed(seed)
    indices = torch.randperm(len(all_pairs))
    data_tensor = data_tensor[indices]
    if frac_train >= 1.0:
        return data_tensor, data_tensor
    div = int(frac_train * len(all_pairs))
    return data_tensor[:div], data_tensor[div:]


# ---------- Style constants ----------
COLORS = ['#0D2758', '#60656F', '#DEA54B', '#A32015', '#347186']
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    'cividis_white_center', ['#0D2758', 'white', '#A32015'], N=256
)
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    'cividis_white_seq', ['white', '#0D2758'], N=256
)
DPI = 150
plt.rcParams['mathtext.fontset'] = 'cm'


def _save_fig(fig, path):
    """Save a figure and close it."""
    fig.savefig(path, dpi=DPI, bbox_inches='tight', format='png')
    plt.close(fig)


# ======================================================================
# Helpers for loading checkpoints
# ======================================================================

def _find_run_dir(base_dir):
    """
    Given a run type directory (e.g. trained_models/p_023/standard/),
    find the actual checkpoint directory.  It may be a timestamped
    subdirectory, or the checkpoints may live directly in base_dir.
    Returns the path that contains the .pth checkpoint files.
    """
    if not os.path.isdir(base_dir):
        return None

    # Check if .pth files live directly here
    pth_files = [f for f in os.listdir(base_dir)
                 if f.endswith('.pth') and f not in ('train_data.pth', 'test_data.pth')]
    if pth_files:
        return base_dir

    # Otherwise look for a single timestamped subdirectory
    subdirs = [d for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]
    for sd in sorted(subdirs):
        candidate = os.path.join(base_dir, sd)
        files = os.listdir(candidate)
        if any(f.endswith('.pth') for f in files):
            return candidate
    return None


def _load_checkpoints(run_dir, device='cpu'):
    """
    Load all numbered checkpoints from run_dir.
    Returns dict {epoch_int: state_dict} sorted by epoch.
    """
    loaded = {}
    exclude = {'final.pth', 'test_data.pth', 'train_data.pth'}
    for fname in os.listdir(run_dir):
        fpath = os.path.join(run_dir, fname)
        if (os.path.isfile(fpath) and fname.endswith('.pth')
                and fname not in exclude):
            try:
                epoch = int(os.path.splitext(fname)[0])
            except ValueError:
                continue
            data = torch.load(fpath, weights_only=True, map_location=device)
            if isinstance(data, dict) and 'model' in data:
                loaded[epoch] = data['model']
            else:
                loaded[epoch] = data
    return {k: loaded[k] for k in sorted(loaded)}


def _load_final(run_dir, device='cpu'):
    """Load the final.pth model data dict."""
    fpath = os.path.join(run_dir, 'final.pth')
    if not os.path.exists(fpath):
        # Fall back to largest epoch checkpoint
        ckpts = _load_checkpoints(run_dir, device)
        if ckpts:
            max_epoch = max(ckpts.keys())
            return {'model': ckpts[max_epoch]}
        return None
    return torch.load(fpath, weights_only=True, map_location=device)


def _load_training_curves(run_type_dir):
    """Load training_curves.json from the run type directory."""
    path = os.path.join(run_type_dir, 'training_curves.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fall back: check inside the checkpoint subdirectory
    run_dir = _find_run_dir(run_type_dir)
    if run_dir and run_dir != run_type_dir:
        path = os.path.join(run_dir, 'training_curves.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    # Fall back: try loading from final.pth
    if run_dir:
        final_path = os.path.join(run_dir, 'final.pth')
        if os.path.exists(final_path):
            data = torch.load(final_path, weights_only=True, map_location='cpu')
            if isinstance(data, dict):
                curves = {}
                for key in ('train_losses', 'test_losses', 'train_accs', 'test_accs',
                            'grad_norms', 'param_norms'):
                    if key in data:
                        val = data[key]
                        if isinstance(val, torch.Tensor):
                            val = val.cpu().tolist()
                        curves[key] = val
                if curves:
                    return curves
    return None


# ======================================================================
# PlotGenerator
# ======================================================================

class PlotGenerator:
    """
    Generates all model-dependent plots for a single prime p.

    Parameters
    ----------
    p : int
        The prime modulus.
    input_dir : str
        Path to trained_models/p_PPP/ containing run-type subdirectories.
    output_dir : str
        Path to hf_app/precomputed_results/p_PPP/ where plots are saved.
    """

    def __init__(self, p, input_dir, output_dir):
        self.p = p
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = 'cpu'
        self.d_vocab = p
        self.d_model = p

        os.makedirs(output_dir, exist_ok=True)

        # Infer d_mlp from checkpoint weights; fall back to formula
        self.d_mlp = self._infer_d_mlp() or compute_d_mlp(p)

        # Fourier basis (mechanism_base version with device arg)
        self.fourier_basis, self.fourier_basis_names = get_fourier_basis(p, self.device)

        # All (a,b) pairs and labels
        self.all_data = torch.tensor(
            [(i, j) for i in range(p) for j in range(p)], dtype=torch.long
        )
        self.all_labels = torch.tensor(
            [(i + j) % p for i in range(p) for j in range(p)], dtype=torch.long
        )

    def _infer_d_mlp(self):
        """Infer d_mlp from the first available checkpoint's weight shape."""
        for run_name in TRAINING_RUNS:
            run_type_dir = os.path.join(self.input_dir, run_name)
            run_dir = _find_run_dir(run_type_dir)
            if run_dir is None:
                continue
            final = _load_final(run_dir, 'cpu')
            if final and 'model' in final and 'mlp.W_in' in final['model']:
                d_mlp = final['model']['mlp.W_in'].shape[0]
                print(f"  Inferred d_mlp={d_mlp} from {run_name} checkpoint")
                return d_mlp
        return None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _run_type_dir(self, run_name):
        return os.path.join(self.input_dir, run_name)

    def _run_dir(self, run_name):
        return _find_run_dir(self._run_type_dir(run_name))

    def _out(self, filename):
        # Prefix every file with pXXX_ so folders are self-contained and browsable
        return os.path.join(self.output_dir, f"p{self.p:03d}_{filename}")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Shared IPR helper
    # ------------------------------------------------------------------

    def _compute_freq_ipr(self, W_dec):
        """IPR over per-frequency magnitudes (combines cos+sin pairs).

        IPR = sum_k A_k^4 / (sum_k A_k^2)^2, where A_k = sqrt(c_k^2 + s_k^2).
        IPR → 1 means all energy at a single frequency.
        Returns mean IPR across neurons.
        """
        K = (self.p - 1) // 2
        A2 = torch.zeros(W_dec.shape[0], K)
        for k in range(1, K + 1):
            A2[:, k - 1] = W_dec[:, 2 * k - 1].pow(2) + W_dec[:, 2 * k].pow(2)
        A4 = A2.pow(2)
        denom = A2.sum(dim=1).pow(2)
        valid = denom > 0
        ipr = torch.zeros(W_dec.shape[0])
        ipr[valid] = A4[valid].sum(dim=1) / denom[valid]
        return ipr.mean()

    def _ipr_at_checkpoint(self, model_sd):
        """Compute average IPR (across both layers) for a single checkpoint."""
        W_in_d, W_out_d, _ = decode_weights(model_sd, self.fourier_basis)
        return ((self._compute_freq_ipr(W_in_d)
                 + self._compute_freq_ipr(W_out_d)) / 2).item()

    # ------------------------------------------------------------------
    # Tab 1: Overview (standard loss+IPR, grokking loss+IPR, phase plot)
    # ------------------------------------------------------------------

    def generate_tab1(self):
        """Generate overview plots: standard + grokking loss/IPR, plus phase scatter."""
        print(f"  [Tab 1] Overview for p={self.p}")

        # ---- Standard run: loss + IPR ----
        std_dir = self._run_dir('standard')
        std_epochs, std_loss, std_ipr = [], [], []
        if std_dir is not None:
            std_curves = _load_training_curves(self._run_type_dir('standard'))
            std_ckpts = _load_checkpoints(std_dir, self.device)
            if std_ckpts:
                std_epochs = sorted(std_ckpts.keys())
                std_ipr = [self._ipr_at_checkpoint(std_ckpts[ep]) for ep in std_epochs]
                if std_curves and 'train_losses' in std_curves:
                    se = std_epochs[1] - std_epochs[0] if len(std_epochs) > 1 else 200
                    std_loss = std_curves['train_losses'][::se][:len(std_epochs)]

        # ---- Grokking run: train/test loss + IPR ----
        grokk_epochs, grokk_train_loss, grokk_test_loss, grokk_ipr = [], [], [], []
        has_grokk = self.p >= MIN_P_GROKKING
        if has_grokk:
            grokk_dir = self._run_dir('grokking')
            if grokk_dir is not None:
                grokk_curves = _load_training_curves(self._run_type_dir('grokking'))
                grokk_ckpts = _load_checkpoints(grokk_dir, self.device)
                if grokk_ckpts:
                    grokk_epochs = sorted(grokk_ckpts.keys())
                    grokk_ipr = [self._ipr_at_checkpoint(grokk_ckpts[ep])
                                 for ep in grokk_epochs]
                    if grokk_curves:
                        se = grokk_epochs[1] - grokk_epochs[0] if len(grokk_epochs) > 1 else 200
                        if 'train_losses' in grokk_curves:
                            grokk_train_loss = grokk_curves['train_losses'][::se][:len(grokk_epochs)]
                        if 'test_losses' in grokk_curves:
                            grokk_test_loss = grokk_curves['test_losses'][::se][:len(grokk_epochs)]

        if not std_epochs and not grokk_epochs:
            print("    SKIP: no checkpoints found for standard or grokking run")
            return

        # ---- Static plot: 2×2 grid (std loss, grokk loss, std IPR, grokk IPR) ----
        n_cols = 2 if has_grokk and grokk_epochs else 1
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 7),
                                 constrained_layout=True)
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        # Standard loss (top-left)
        ax = axes[0, 0]
        if std_loss:
            ax.plot(std_epochs[:len(std_loss)], std_loss,
                    color=COLORS[0], linewidth=1.5, label="Train Loss")
        ax.set_title('Standard (ReLU, full data)', fontsize=14)
        ax.set_ylabel('Loss', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4)

        # Standard IPR (bottom-left)
        ax = axes[1, 0]
        if std_ipr:
            ax.plot(std_epochs[:len(std_ipr)], std_ipr,
                    color=COLORS[3], linewidth=1.5, label="Avg. IPR")
        ax.set_title('Standard IPR', fontsize=14)
        ax.set_xlabel('Step', fontsize=13)
        ax.set_ylabel('IPR', fontsize=13)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4)

        if n_cols == 2:
            # Grokking loss (top-right)
            ax = axes[0, 1]
            gx = grokk_epochs
            if grokk_train_loss:
                ax.plot(gx[:len(grokk_train_loss)], grokk_train_loss,
                        color=COLORS[0], linewidth=1.5, label="Train Loss")
            if grokk_test_loss:
                ax.plot(gx[:len(grokk_test_loss)], grokk_test_loss,
                        color=COLORS[3], linewidth=1.5, label="Test Loss")
            ax.set_title('Grokking (ReLU, 75% data, WD)', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.4)

            # Grokking IPR (bottom-right)
            ax = axes[1, 1]
            if grokk_ipr:
                ax.plot(gx[:len(grokk_ipr)], grokk_ipr,
                        color=COLORS[3], linewidth=1.5, label="Avg. IPR")
            ax.set_title('Grokking IPR', fontsize=14)
            ax.set_xlabel('Step', fontsize=13)
            ax.set_ylim([0, 1.05])
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.4)

        _save_fig(fig, self._out('overview_loss_ipr.png'))

        # ---- Phase relationship scatter from standard final checkpoint ----
        if std_ckpts:
            final_ep = max(std_ckpts.keys())
            model_sd = std_ckpts[final_ep]
            W_in_d, W_out_d, mfl = decode_weights(model_sd, self.fourier_basis)
            n_neurons = W_in_d.shape[0]
            phis_2, psis = [], []
            for neuron in range(n_neurons):
                _, phi = compute_neuron(neuron, mfl, W_in_d)
                _, psi = compute_neuron(neuron, mfl, W_out_d)
                two_phi = normalize_to_pi(2 * phi)
                psi_n = normalize_to_pi(psi)
                # Fix ±π wrap: keep ψ within π of 2φ
                if psi_n - two_phi > np.pi:
                    psi_n -= 2 * np.pi
                elif psi_n - two_phi < -np.pi:
                    psi_n += 2 * np.pi
                phis_2.append(two_phi)
                psis.append(psi_n)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r-',
                    linewidth=3, alpha=0.8,
                    label=r'$\psi_m = 2\phi_m$', zorder=1)
            ax.scatter(phis_2, psis, s=12, alpha=0.6, color=COLORS[0], zorder=2)
            ax.legend(fontsize=12, loc='upper left')
            ax.set_xlabel(r'$2\phi_m$', fontsize=14)
            ax.set_ylabel(r'$\psi_m$', fontsize=14)
            ax.set_title(r'Phase Alignment: $\psi_m = 2\phi_m$', fontsize=14)
            ax.set_xlim([-np.pi, np.pi])
            ax.set_ylim([-np.pi, np.pi])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            _save_fig(fig, self._out('overview_phase_scatter.png'))

        # ---- JSON for interactive Plotly charts ----
        payload = {
            'std_epochs': [int(e) for e in std_epochs],
            'std_ipr': std_ipr,
        }
        if std_loss:
            payload['std_train_loss'] = [float(v) for v in std_loss]

        if has_grokk and grokk_epochs:
            payload['grokk_epochs'] = [int(e) for e in grokk_epochs]
            payload['grokk_ipr'] = grokk_ipr
            if grokk_train_loss:
                payload['grokk_train_loss'] = [float(v) for v in grokk_train_loss]
            if grokk_test_loss:
                payload['grokk_test_loss'] = [float(v) for v in grokk_test_loss]

        with open(self._out('overview.json'), 'w') as f:
            json.dump(payload, f)

        files = ['overview_loss_ipr.png', 'overview.json']
        if std_ckpts:
            files.append('overview_phase_scatter.png')
        print(f"    Saved {', '.join(files)}")

    # ------------------------------------------------------------------
    # Tab 2: Fourier Weights (heatmap + lineplots)
    # ------------------------------------------------------------------

    def generate_tab2(self):
        """Generate dft_heatmap_in.png, dft_heatmap_out.png, lineplot_in.png, lineplot_out.png."""
        print(f"  [Tab 2] Fourier Weights for p={self.p}")
        run_dir = self._run_dir('standard')
        if run_dir is None:
            print("    SKIP: standard run directory not found")
            return

        final_data = _load_final(run_dir, self.device)
        if final_data is None:
            print("    SKIP: no final checkpoint")
            return
        model_load = final_data['model']

        W_in_decode, W_out_decode, max_freq_ls = decode_weights(
            model_load, self.fourier_basis
        )
        d_mlp = W_in_decode.shape[0]
        num_neurons = min(20, d_mlp)

        # Sort neurons by frequency
        sorted_indices = select_top_neurons_by_frequency(
            max_freq_ls, W_in_decode, n=num_neurons
        )

        freq_ls = np.array([max_freq_ls[i] for i in sorted_indices])

        # DFT coefficients for heatmap (matches blog Figure 2)
        W_in_dft = W_in_decode[sorted_indices, :]
        W_out_dft = W_out_decode[sorted_indices, :]
        # Raw weights for line plots (matches blog Figure 3)
        W_in_raw = model_load['mlp.W_in'][sorted_indices, :]
        W_out_raw = model_load['mlp.W_out'].T[sorted_indices, :]

        # Sort within selected set by frequency
        sort_order = np.argsort(freq_ls)
        ranked_W_in_dft = W_in_dft[sort_order, :]
        ranked_W_out_dft = W_out_dft[sort_order, :]
        ranked_W_in_raw = W_in_raw[sort_order, :]
        ranked_W_out_raw = W_out_raw[sort_order, :]

        # ---- Heatmap plots (DFT coefficients, matching blog Figure 2) ----
        # Save as two separate images for side-by-side display in the app
        fb_names = self.fourier_basis_names
        n_modes = len(fb_names)
        fig_w = max(6, n_modes * 0.35)
        fig_h = max(5, num_neurons * 0.3 + 2)
        y_locs = np.arange(num_neurons)
        x_locs = np.arange(n_modes)

        # W_in DFT (first-layer / W_E)
        W_in_np = ranked_W_in_dft.detach().cpu().numpy()
        abs_max_in = np.abs(W_in_np).max()
        fig_in, ax_in = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        im_in = ax_in.imshow(
            W_in_np,
            cmap=CMAP_DIVERGING, vmin=-abs_max_in, vmax=abs_max_in,
            aspect='auto'
        )
        ax_in.set_title(r'First-Layer $\theta_m$ (W$_E$) after DFT', fontsize=16)
        fig_in.colorbar(im_in, ax=ax_in, shrink=0.8)
        ax_in.set_yticks(y_locs)
        ax_in.set_yticklabels(y_locs, fontsize=10)
        ax_in.set_ylabel('Neuron #', fontsize=14)
        ax_in.set_xticks(x_locs)
        ax_in.set_xticklabels(fb_names, rotation=90, fontsize=10)
        ax_in.set_xlabel('Fourier Component', fontsize=14)
        _save_fig(fig_in, self._out('dft_heatmap_in.png'))

        # W_out DFT (second-layer / W_L)
        W_out_np = ranked_W_out_dft.detach().cpu().numpy()
        abs_max_out = np.abs(W_out_np).max()
        fig_out, ax_out = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        im_out = ax_out.imshow(
            W_out_np,
            cmap=CMAP_DIVERGING, vmin=-abs_max_out, vmax=abs_max_out,
            aspect='auto'
        )
        ax_out.set_title(r'Second-Layer $\xi_m$ (W$_L$) after DFT', fontsize=16)
        fig_out.colorbar(im_out, ax=ax_out, shrink=0.8)
        ax_out.set_yticks(y_locs)
        ax_out.set_yticklabels(y_locs, fontsize=10)
        ax_out.set_ylabel('Neuron #', fontsize=14)
        ax_out.set_xticks(x_locs)
        ax_out.set_xticklabels(fb_names, rotation=90, fontsize=10)
        ax_out.set_xlabel('Fourier Component', fontsize=14)
        _save_fig(fig_out, self._out('dft_heatmap_out.png'))

        # ---- Line plots (raw weights + cosine fits, matching blog Figure 3) ----
        lineplot_idx = select_lineplot_neurons(list(range(num_neurons)), n=3)
        fb = self.fourier_basis
        positions = np.arange(ranked_W_in_raw.shape[1])

        for tag, weight_data, title_tex in [
            ('lineplot_in', ranked_W_in_raw, r'First-Layer Parameters $\theta_m$'),
            ('lineplot_out', ranked_W_out_raw, r'Second-Layer Parameters $\xi_m$'),
        ]:
            if hasattr(weight_data, 'detach'):
                weight_np = weight_data.detach().cpu()
            else:
                weight_np = weight_data

            top3 = weight_np[lineplot_idx]

            lp_w = max(8, self.p * 0.35)
            fig, axes_lp = plt.subplots(
                nrows=3, ncols=1, figsize=(lp_w, 8),
                constrained_layout=True,
                gridspec_kw={'hspace': 0.08}
            )

            for i, ax in enumerate(axes_lp):
                data = top3[i]
                if isinstance(data, torch.Tensor):
                    data_t = data.float()
                else:
                    data_t = torch.tensor(data, dtype=torch.float32)
                # Project into Fourier space, keep top 2 components, project back
                proj = data_t @ fb.T
                abs_proj = torch.abs(proj)
                _, top2_idx = torch.topk(abs_proj, 2)
                mask = torch.zeros_like(proj)
                mask[top2_idx] = proj[top2_idx]
                data_est = mask @ fb
                data_np = data_t.numpy()
                data_est_np = data_est.numpy()

                ax.plot(data_np, marker='o', markersize=5,
                        color=COLORS[0], linewidth=1.5, linestyle=':',
                        label="Actual")
                ax.plot(data_est_np, marker='o', markersize=5,
                        color=COLORS[3], linewidth=1.5, linestyle=':',
                        alpha=0.7, label="Fitted")
                ax.set_ylim(-0.9, 0.9)
                ax.set_ylabel(f'Neuron #{i+1}', fontsize=14)
                ax.set_xticks(positions)
                ax.grid(True, which='major', axis='both',
                        linestyle='--', linewidth=0.5, alpha=0.6)
                if i < len(axes_lp) - 1:
                    ax.set_xticklabels([])
                ax.legend(fontsize=12, loc="upper right")

            axes_lp[-1].set_xlabel('Input Dimension', fontsize=14)
            axes_lp[-1].set_xticks(positions)
            axes_lp[-1].set_xticklabels(
                np.arange(ranked_W_in_raw.shape[1]), rotation=0, fontsize=10
            )
            axes_lp[0].set_title(title_tex, fontsize=18)

            _save_fig(fig, self._out(f'{tag}.png'))

        print("    Saved dft_heatmap_in.png, dft_heatmap_out.png, lineplot_in.png, lineplot_out.png")

    # ------------------------------------------------------------------
    # Tab 3: Phase Analysis
    # ------------------------------------------------------------------

    def generate_tab3(self):
        """Generate phase_distribution.png, phase_relationship.png, magnitude_distribution.png."""
        print(f"  [Tab 3] Phase Analysis for p={self.p}")
        run_dir = self._run_dir('standard')
        if run_dir is None:
            print("    SKIP: standard run directory not found")
            return

        final_data = _load_final(run_dir, self.device)
        if final_data is None:
            print("    SKIP: no final checkpoint")
            return
        model_load = final_data['model']

        W_in_decode, W_out_decode, max_freq_ls = decode_weights(
            model_load, self.fourier_basis
        )
        d_mlp = W_in_decode.shape[0]

        # Compute all neuron phases and magnitudes
        coeff_in_scale_ls = []
        coeff_out_scale_ls = []
        coeff_phi_ls = []
        coeff_psi_ls = []

        for neuron in range(d_mlp):
            s_in, phi_in = compute_neuron(neuron, max_freq_ls, W_in_decode)
            s_out, phi_out = compute_neuron(neuron, max_freq_ls, W_out_decode)
            coeff_in_scale_ls.append(s_in)
            coeff_out_scale_ls.append(s_out)
            coeff_phi_ls.append(phi_in)
            coeff_psi_ls.append(phi_out)

        coeff_phi_arr = np.array(coeff_phi_ls)
        coeff_psi_arr = np.array(coeff_psi_ls)

        # ---- Phase distribution on concentric circles ----
        # Select the most common non-zero frequency for phase analysis
        target_freq = select_phase_frequency(max_freq_ls, self.p)
        freq_neurons = [i for i, f in enumerate(max_freq_ls) if f == target_freq]
        phi_subset = np.array([coeff_phi_ls[i] for i in freq_neurons])

        theta = np.linspace(0, 2 * np.pi, 300)
        multipliers = [1, 2, 3, 4]
        radii = [1.0, 0.88, 0.76, 0.64]

        fig, ax = plt.subplots(figsize=(4, 4))
        for m, r in zip(multipliers, radii):
            x_c, y_c = r * np.cos(theta), r * np.sin(theta)
            ax.plot(x_c, y_c, linewidth=0.8, color='gray', alpha=0.6)

            x_pts = r * np.cos(m * phi_subset)
            y_pts = r * np.sin(m * phi_subset)
            label = fr'$\phi_m$' if m == 1 else fr'${m}\phi_m$'
            ax.scatter(x_pts, y_pts, s=20, marker='o',
                       color=COLORS[m - 1], label=label)

        ax.legend(
            fontsize=15, loc='upper center', columnspacing=0.2,
            handletextpad=0.1, bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False
        )
        ax.set_xlabel(r'$\cos(\phi_m)$', fontsize=19)
        ax.set_ylabel(r'$\sin(\phi_m)$', fontsize=19)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        _save_fig(fig, self._out('phase_distribution.png'))

        # ---- Phase relationship: 2*phi vs psi ----
        coeff_2phi_arr = np.array([normalize_to_pi(2 * phi) for phi in coeff_phi_arr])
        coeff_psi_plot = coeff_psi_arr.copy()
        # Fix ±π wrap: keep ψ within π of 2φ so boundary points stay on diagonal
        diff = coeff_psi_plot - coeff_2phi_arr
        coeff_psi_plot[diff > np.pi] -= 2 * np.pi
        coeff_psi_plot[diff < -np.pi] += 2 * np.pi

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r-', linewidth=3, alpha=0.8,
                label=r'$\psi_m = 2\phi_m$', zorder=1)
        ax.scatter(
            coeff_2phi_arr, coeff_psi_plot,
            marker='.', color=COLORS[0], s=20, zorder=2
        )
        ax.legend(fontsize=12, loc='upper left')
        ax.set_xlabel(r'$2\phi_m$', fontsize=14)
        ax.set_ylabel(r'$\psi_m$', fontsize=14)
        ax.set_title(r'Phase Alignment: $\psi_m = 2\phi_m$', fontsize=14)
        ax.set_xlim(-np.pi * 1.1, np.pi * 1.1)
        ax.set_ylim(-np.pi * 1.1, np.pi * 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        _save_fig(fig, self._out('phase_relationship.png'))

        # ---- Magnitude distribution (violin) ----
        fig, ax = plt.subplots(figsize=(4, 4))
        data_for_plot = [coeff_in_scale_ls, coeff_out_scale_ls]
        positions = [1, 2]

        parts = ax.violinplot(
            data_for_plot, positions=positions, widths=0.6,
            showmeans=True, showmedians=True, showextrema=True
        )
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS[0])
            pc.set_alpha(0.7)
        parts['cmedians'].set_color(COLORS[2])
        parts['cmedians'].set_linewidth(2)
        parts['cmeans'].set_color(COLORS[2])
        parts['cmeans'].set_linewidth(2)
        parts['cbars'].set_color(COLORS[0])
        parts['cbars'].set_linewidth(1.5)
        parts['cmaxes'].set_color(COLORS[0])
        parts['cmins'].set_color(COLORS[0])

        ax.set_xticks(positions)
        ax.set_xticklabels(['First-Layer', 'Second-Layer'], fontsize=14)
        ax.set_ylabel('Magnitude', fontsize=19)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        _save_fig(fig, self._out('magnitude_distribution.png'))

        print("    Saved phase_distribution.png, phase_relationship.png, magnitude_distribution.png")

    # ------------------------------------------------------------------
    # Tab 4: Output Logits
    # ------------------------------------------------------------------

    def generate_tab4(self):
        """Generate output_logits.png."""
        print(f"  [Tab 4] Output Logits for p={self.p}")
        run_dir = self._run_dir('standard')
        if run_dir is None:
            print("    SKIP: standard run directory not found")
            return

        final_data = _load_final(run_dir, self.device)
        if final_data is None:
            print("    SKIP: no final checkpoint")
            return
        model_load = final_data['model']

        p = self.p
        act_type = TRAINING_RUNS['standard']['act_type']
        model = EmbedMLP(
            d_vocab=self.d_vocab,
            d_model=self.d_model,
            d_mlp=self.d_mlp,
            act_type=act_type,
            use_cache=False
        )
        model.to(self.device)
        model.load_state_dict(model_load)
        model.eval()

        with torch.no_grad():
            logits = model(self.all_data).squeeze(1)

        logits_np = logits.cpu().numpy()

        # Show first p pairs (first row of the input grid)
        interval_start = 0
        interval_end = p
        logits_interval = logits_np[interval_start:interval_end]
        selected_pairs = self.all_data[interval_start:interval_end]

        fig, ax = plt.subplots(figsize=(7, 6))
        abs_max = np.abs(logits_np).max() * 0.8
        im = ax.imshow(
            logits_interval.T, cmap=CMAP_DIVERGING, aspect='auto',
            vmin=-abs_max, vmax=abs_max
        )

        # Highlight target positions with rectangles
        for i, (x_val_t, y_val_t) in enumerate(selected_pairs):
            x_val = x_val_t.item()
            y_val = y_val_t.item()
            target_2x = (2 * x_val) % p
            target_2y = (2 * y_val) % p
            target_sum = (x_val + y_val) % p

            rect_2x = patches.Rectangle(
                (i - 0.5, target_2x - 0.5), 1, 1,
                linewidth=1.6, edgecolor='#0D2758', facecolor='none', alpha=0.9
            )
            ax.add_patch(rect_2x)
            if target_2y != target_2x:
                rect_2y = patches.Rectangle(
                    (i - 0.5, target_2y - 0.5), 1, 1,
                    linewidth=1.6, edgecolor='#0D2758', facecolor='none', alpha=0.9
                )
                ax.add_patch(rect_2y)
            rect_sum = patches.Rectangle(
                (i - 0.5, target_sum - 0.5), 1, 1,
                linewidth=1.6, edgecolor='#0D2758', facecolor='none', alpha=0.9
            )
            ax.add_patch(rect_sum)

        n_pairs = interval_end - interval_start
        if n_pairs <= 50:
            x_positions = np.arange(n_pairs)
            x_labels = [f"({selected_pairs[i][0].item()},{selected_pairs[i][1].item()})"
                        for i in range(n_pairs)]
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=14)
        else:
            n_labels = min(25, n_pairs)
            step = n_pairs // n_labels
            x_positions = np.arange(0, n_pairs, step)
            x_labels = [f"({selected_pairs[i][0].item()},{selected_pairs[i][1].item()})"
                        for i in x_positions]
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=90, ha='right', fontsize=14)

        ax.set_yticks(np.arange(p))
        ax.set_yticklabels(np.arange(p), fontsize=14)
        ax.set_xlabel("Input Pair", fontsize=18)
        ax.set_ylabel("Output", fontsize=18)
        plt.colorbar(im, ax=ax)
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, axis='x')
        plt.tight_layout()

        _save_fig(fig, self._out('output_logits.png'))
        print("    Saved output_logits.png")

    # ------------------------------------------------------------------
    # Tab 5: Grokking
    # ------------------------------------------------------------------

    def generate_tab5(self):
        """Generate grokking-related plots."""
        print(f"  [Tab 5] Grokking for p={self.p}")
        if self.p < MIN_P_GROKKING:
            print(f"    SKIP: p={self.p} < {MIN_P_GROKKING} (too few test points for grokking)")
            return
        run_dir = self._run_dir('grokking')
        if run_dir is None:
            print("    SKIP: grokking run directory not found")
            return

        curves = _load_training_curves(self._run_type_dir('grokking'))
        checkpoints = _load_checkpoints(run_dir, self.device)

        if not checkpoints:
            print("    SKIP: no grokking checkpoints")
            return

        epochs = sorted(checkpoints.keys())
        p = self.p
        d_mlp = self.d_mlp
        act_type = TRAINING_RUNS['grokking']['act_type']

        # Load train/test data
        train_data_path = os.path.join(run_dir, 'train_data.pth')
        test_data_path = os.path.join(run_dir, 'test_data.pth')
        train_data = None
        test_data = None
        train_labels = None
        test_labels = None
        if os.path.exists(train_data_path):
            raw = torch.load(train_data_path, weights_only=False,
                             map_location=self.device)
            # Handle both formats: plain tensor or (pairs, labels) tuple
            if isinstance(raw, (tuple, list)):
                train_data, train_labels = raw[0], raw[1]
            else:
                train_data = raw
        if os.path.exists(test_data_path):
            raw = torch.load(test_data_path, weights_only=False,
                             map_location=self.device)
            if isinstance(raw, (tuple, list)):
                test_data, test_labels = raw[0], raw[1]
            else:
                test_data = raw

        # Fallback: regenerate data deterministically if files are missing
        if train_data is None or test_data is None:
            grokk_cfg = TRAINING_RUNS['grokking']
            frac = grokk_cfg['frac_train']
            seed = grokk_cfg['seed']
            print(f"    Regenerating train/test data (frac={frac}, seed={seed})")
            train_data, test_data = _gen_train_test(p, frac_train=frac, seed=seed)

        # Compute labels from pairs if not loaded directly
        if train_labels is None and train_data is not None:
            train_labels = torch.tensor(
                [(train_data[i, 0].item() + train_data[i, 1].item()) % p
                 for i in range(train_data.shape[0])],
                dtype=torch.long
            )
        if test_labels is None and test_data is not None:
            test_labels = torch.tensor(
                [(test_data[i, 0].item() + test_data[i, 1].item()) % p
                 for i in range(test_data.shape[0])],
                dtype=torch.long
            )

        # Detect stage boundaries
        train_losses = curves.get('train_losses', []) if curves else []
        test_losses = curves.get('test_losses', []) if curves else []
        train_accs_curve = curves.get('train_accs', None) if curves else None
        test_accs_curve = curves.get('test_accs', None) if curves else None

        stage1_end, stage2_end = detect_grokking_stages(
            train_losses, test_losses, train_accs_curve, test_accs_curve
        )
        if stage1_end is None:
            stage1_end = len(epochs) // 5
        if stage2_end is None:
            stage2_end = len(epochs) * 3 // 5

        # ---- Loss JSON + static PNG ----
        if train_losses:
            loss_data = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'stage1_end': stage1_end,
                'stage2_end': stage2_end,
            }
            with open(self._out('grokk_loss.json'), 'w') as f:
                json.dump(loss_data, f)

            # Static loss PNG (matches blog Figure 13a)
            max_step = min(len(train_losses), len(test_losses)) if test_losses else len(train_losses)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot(train_losses[:max_step], color='#0D2758', linewidth=2, label='Train')
            if test_losses:
                ax.plot(test_losses[:max_step], color='#A32015', linewidth=2, label='Test')
            ax.axvspan(0, stage1_end, alpha=0.15, color='#D4AF37')
            ax.axvspan(stage1_end, stage2_end, alpha=0.15, color='#8B7355')
            ax.axvspan(stage2_end, max_step, alpha=0.15, color='#60656F')
            ax.axvline(x=stage1_end, color='black', linestyle='--', linewidth=1)
            ax.axvline(x=stage2_end, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Step', fontsize=16)
            ax.set_ylabel('Loss', fontsize=16)
            ax.legend(fontsize=16, loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            _save_fig(fig, self._out('grokk_loss.png'))

        # ---- Accuracy: compute from checkpoints if not in curves ----
        train_accs = []
        test_accs = []
        if train_data is not None and test_data is not None:
            for ep in epochs:
                model = EmbedMLP(
                    d_vocab=self.d_vocab, d_model=self.d_model,
                    d_mlp=d_mlp, act_type=act_type, use_cache=False
                ).to(self.device)
                model.load_state_dict(checkpoints[ep])
                model.eval()
                with torch.no_grad():
                    tr_logits = model(train_data)
                    te_logits = model(test_data)
                train_accs.append(acc_rate(tr_logits, train_labels))
                test_accs.append(acc_rate(te_logits, test_labels))
        elif train_accs_curve is not None:
            # Use curves data, subsample to match checkpoint epochs
            save_every = epochs[1] - epochs[0] if len(epochs) > 1 else 200
            train_accs = train_accs_curve[::save_every][:len(epochs)]
            test_accs = test_accs_curve[::save_every][:len(epochs)]

        acc_data = {
            'epochs': epochs,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'stage1_end': stage1_end,
            'stage2_end': stage2_end,
        }
        with open(self._out('grokk_acc.json'), 'w') as f:
            json.dump(acc_data, f)

        # Static accuracy PNG (matches blog Figure 13b)
        if train_accs and test_accs:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axvspan(0, stage1_end, alpha=0.15, color='#D4AF37')
            ax.axvspan(stage1_end, stage2_end, alpha=0.15, color='#8B7355')
            ax.axvspan(stage2_end, epochs[-1] if epochs else stage2_end,
                       alpha=0.15, color='#60656F')
            ax.axvline(x=stage1_end, color='black', linestyle='--', linewidth=1)
            ax.axvline(x=stage2_end, color='black', linestyle='--', linewidth=1)
            ax.plot(epochs[:len(train_accs)], train_accs,
                    label='Train', color='#0D2758', linewidth=2.5)
            ax.plot(epochs[:len(test_accs)], test_accs,
                    label='Test', color='#A32015', linewidth=2.5)
            ax.set_xlabel('Step', fontsize=16)
            ax.set_ylabel('Accuracy', fontsize=16)
            ax.legend(fontsize=16, loc='lower right')
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            _save_fig(fig, self._out('grokk_acc.png'))

        # ---- Phase difference |sin(D*)| ----
        abs_phase_diff = []
        sparse_level = []

        for ep in epochs:
            model_sd = checkpoints[ep]
            W_in_d, W_out_d, mfl = decode_weights(model_sd, self.fourier_basis)

            sparse_level.append(self._ipr_at_checkpoint(model_sd))

            phase_diffs = []
            for neuron in range(W_in_d.shape[0]):
                _, phi_in = compute_neuron(neuron, mfl, W_in_d)
                _, phi_out = compute_neuron(neuron, mfl, W_out_d)
                phase_diffs.append(normalize_to_pi(phi_out - 2 * phi_in))
            phase_diffs = np.array(phase_diffs)
            abs_phase_diff.append(np.mean(np.abs(np.sin(phase_diffs))))

        # Limit to reasonable number of points for plotting
        n_plot = min(len(epochs), 100)
        x_phase = np.array(epochs[:n_plot])

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axvspan(0, stage1_end, alpha=0.15, color='#D4AF37')
        ax.axvspan(stage1_end, min(stage2_end, x_phase[-1] if len(x_phase) else stage2_end),
                   alpha=0.15, color='#8B7355')
        if len(x_phase):
            ax.axvspan(stage2_end, x_phase[-1], alpha=0.15, color='#60656F')
        ax.axvline(x=stage1_end, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=stage2_end, color='black', linestyle='--', linewidth=1)
        ax.plot(x_phase, abs_phase_diff[:n_plot], marker='x', markersize=5,
                color='#986d56', label=r"Avg. $|\sin(D_m^\star)|$", linewidth=1.5)
        ax.set_xlabel('Step', fontsize=16)
        ax.set_ylabel('Average Value', fontsize=16)
        ax.set_ylim([0, 0.65])
        ax.legend(fontsize=16, loc="upper right")
        ax.grid(True, alpha=0.5, linestyle='--')
        plt.tight_layout()
        _save_fig(fig, self._out('grokk_abs_phase_diff.png'))

        # ---- IPR + param norms (dual axis) ----
        x_all = np.array(epochs)
        param_norms = []
        if curves and 'param_norms' in curves:
            save_every = epochs[1] - epochs[0] if len(epochs) > 1 else 200
            param_norms = curves['param_norms'][::save_every][:len(epochs)]

        fig, ax1 = plt.subplots(figsize=(4, 4))
        ax1.axvspan(0, stage1_end, alpha=0.15, color='#D4AF37')
        ax1.axvspan(stage1_end, min(stage2_end, x_all[-1] if len(x_all) else stage2_end),
                    alpha=0.15, color='#8B7355')
        if len(x_all):
            ax1.axvspan(stage2_end, x_all[-1], alpha=0.15, color='#60656F')
        ax1.axvline(x=stage1_end, color='black', linestyle='--', linewidth=1)
        ax1.axvline(x=stage2_end, color='black', linestyle='--', linewidth=1)

        line1 = ax1.plot(x_all, sparse_level, marker='x', markersize=3,
                         color='#986d56', label=r"Avg. IPR", linewidth=1.5)
        ax1.set_xlabel('Step', fontsize=16)
        ax1.tick_params(axis='y')
        ax1.set_ylim([0, 1.05])

        if param_norms:
            ax2 = ax1.twinx()
            line2 = ax2.plot(x_all[:len(param_norms)], param_norms,
                             marker='o', markersize=3, color='#2E5266',
                             label=r"Param. Norm", linewidth=1.5)
            ax2.tick_params(axis='y')
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, fontsize=16, loc="lower right")
        else:
            ax1.legend(fontsize=16, loc="lower right")

        ax1.grid(True, alpha=0.5, linestyle='--')
        plt.tight_layout()
        _save_fig(fig, self._out('grokk_avg_ipr.png'))

        # ---- Memorization accuracy (3-panel) ----
        if train_data is not None:
            # Find a checkpoint near stage1_end
            closest_epoch = min(epochs, key=lambda e: abs(e - stage1_end))
            model_sd = checkpoints[closest_epoch]

            model = EmbedMLP(
                d_vocab=self.d_vocab, d_model=self.d_model,
                d_mlp=d_mlp, act_type=act_type, use_cache=False
            ).to(self.device)
            model.load_state_dict(model_sd)
            model.eval()

            with torch.no_grad():
                logits = model(self.all_data).squeeze(1)

            train_set = set([(int(i), int(j)) for i, j in train_data])
            true_test_points = []

            train_mask = torch.zeros(p, p)
            for i in range(p):
                for j in range(p):
                    if (i, j) in train_set:
                        train_mask[i, j] = 1.0
                    elif (j, i) in train_set:
                        train_mask[i, j] = 0.65
                    else:
                        train_mask[i, j] = 0.0
                        true_test_points.append((i, j))

            predicted = torch.argmax(logits, dim=1).view(p, p)
            gt_grid = self.all_labels.view(p, p)
            accuracy_mask = (predicted == gt_grid).float()

            probs = torch.softmax(logits, dim=1)
            gt_probs = torch.zeros(p * p)
            for idx in range(p * p):
                i_val = self.all_data[idx, 0].item()
                j_val = self.all_data[idx, 1].item()
                correct = (i_val + j_val) % p
                gt_probs[idx] = probs[idx, correct]
            gt_probs_grid = gt_probs.view(p, p)

            fig = plt.figure(figsize=(20, 6))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.15)

            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])

            # Train mask
            im1 = ax1.imshow(train_mask.numpy(), cmap=CMAP_SEQUENTIAL,
                             vmin=0, vmax=1, aspect='equal')
            ax1.set_title('Training Data under Symmetry', fontsize=21)
            ax1.set_ylabel('First Input', fontsize=18)
            ax1.set_xlabel('Second Input', fontsize=18)
            locs = np.arange(p)
            ax1.set_xticks(locs)
            ax1.set_yticks(locs)
            ax1.set_xticklabels(locs, fontsize=11)
            ax1.set_yticklabels(locs, fontsize=11)
            for ti, tj in true_test_points:
                rect = plt.Rectangle((tj - 0.5, ti - 0.5), 1, 1,
                                     linewidth=2.5, edgecolor='red', facecolor='none')
                ax1.add_patch(rect)

            # Accuracy mask
            im2 = ax2.imshow(accuracy_mask.numpy(), cmap=CMAP_SEQUENTIAL,
                             vmin=0, vmax=1, aspect='equal')
            ax2.set_title('Accuracy before Grokking', fontsize=21)
            ax2.set_xlabel('Second Input', fontsize=18)
            ax2.set_xticks(locs)
            ax2.set_yticks(locs)
            ax2.set_xticklabels(locs, fontsize=11)
            ax2.set_yticklabels(locs, fontsize=11)
            for ti, tj in true_test_points:
                rect = plt.Rectangle((tj - 0.5, ti - 0.5), 1, 1,
                                     linewidth=2.5, edgecolor='red', facecolor='none')
                ax2.add_patch(rect)

            # Softmax probability
            prob_max = gt_probs_grid.max().item()
            im3 = ax3.imshow(gt_probs_grid.detach().numpy(), cmap=CMAP_SEQUENTIAL,
                             vmin=0, vmax=prob_max, aspect='equal')
            ax3.set_title('Softmax Weight at Ground-Truth', fontsize=21)
            ax3.set_xlabel('Second Input', fontsize=18)
            ax3.set_xticks(locs)
            ax3.set_yticks(locs)
            ax3.set_xticklabels(locs, fontsize=11)
            ax3.set_yticklabels(locs, fontsize=11)
            for ti, tj in true_test_points:
                rect = plt.Rectangle((tj - 0.5, ti - 0.5), 1, 1,
                                     linewidth=2.5, edgecolor='red', facecolor='none')
                ax3.add_patch(rect)
            cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.ax.tick_params(labelsize=12)
            plt.tight_layout()
            _save_fig(fig, self._out('grokk_memorization_accuracy.png'))

        # ---- Memorization common-to-rare (4-panel) ----
        if train_data is not None:
            train_set = set([(int(i), int(j)) for i, j in train_data])
            asymmetric_train_points = []
            train_mask_dist = torch.zeros(p, p)
            for i in range(p):
                for j in range(p):
                    if (i, j) in train_set and (j, i) in train_set:
                        train_mask_dist[i, j] = 1.0
                    elif (i, j) in train_set and (j, i) not in train_set:
                        train_mask_dist[i, j] = 0.5
                        asymmetric_train_points.append((i, j))
                    else:
                        train_mask_dist[i, j] = 0.0

            # Pick 3 epochs: 0, ~stage1/2, ~stage1
            selected_epochs = [0]
            mid_epoch = min(epochs, key=lambda e: abs(e - stage1_end // 2))
            end_epoch = min(epochs, key=lambda e: abs(e - stage1_end))
            if mid_epoch not in selected_epochs:
                selected_epochs.append(mid_epoch)
            if end_epoch not in selected_epochs:
                selected_epochs.append(end_epoch)
            # Ensure we have exactly 3 + distribution = 4 panels
            while len(selected_epochs) < 3:
                selected_epochs.append(epochs[min(len(epochs) - 1, 2)])

            fig = plt.figure(figsize=(26, 6))
            gs = fig.add_gridspec(
                1, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.15
            )

            # Panel 1: training data distribution
            ax_d = fig.add_subplot(gs[0])
            ax_d.imshow(train_mask_dist.numpy(), cmap=CMAP_SEQUENTIAL,
                        vmin=0, vmax=1, aspect='equal')
            ax_d.set_title('Training Data Distribution', fontsize=21)
            ax_d.set_ylabel('First Input', fontsize=18)
            ax_d.set_xlabel('Second Input', fontsize=18)
            locs = np.arange(p)
            ax_d.set_xticks(locs)
            ax_d.set_yticks(locs)
            ax_d.set_xticklabels(locs, fontsize=11)
            ax_d.set_yticklabels(locs, fontsize=11)
            for ti, tj in asymmetric_train_points:
                rect = plt.Rectangle((tj - 0.5, ti - 0.5), 1, 1,
                                     linewidth=2.0, edgecolor='red', facecolor='none')
                ax_d.add_patch(rect)

            # Panels 2-4: accuracy at selected epochs
            for panel_idx, sel_ep in enumerate(selected_epochs):
                ax_p = fig.add_subplot(gs[panel_idx + 1])
                model_p = EmbedMLP(
                    d_vocab=self.d_vocab, d_model=self.d_model,
                    d_mlp=d_mlp, act_type=act_type, use_cache=False
                ).to(self.device)
                model_p.load_state_dict(checkpoints[sel_ep])
                model_p.eval()
                with torch.no_grad():
                    logits_p = model_p(self.all_data).squeeze(1)
                pred_p = torch.argmax(logits_p, dim=1).view(p, p)
                acc_p = (pred_p == gt_grid).float()

                ax_p.imshow(acc_p.numpy(), cmap=CMAP_SEQUENTIAL,
                            vmin=0, vmax=1, aspect='equal')
                ax_p.set_title(f'Accuracy at Step {sel_ep}', fontsize=21)
                ax_p.set_xlabel('Second Input', fontsize=18)
                ax_p.set_xticks(locs)
                ax_p.set_yticks(locs)
                ax_p.set_xticklabels(locs, fontsize=11)
                ax_p.set_yticklabels(locs, fontsize=11)
                for ti, tj in asymmetric_train_points:
                    rect = plt.Rectangle((tj - 0.5, ti - 0.5), 1, 1,
                                         linewidth=2.0, edgecolor='red', facecolor='none')
                    ax_p.add_patch(rect)

            plt.tight_layout()
            _save_fig(fig, self._out('grokk_memorization_common_to_rare.png'))

        # ---- Decoded weights dynamic (3 timepoints) ----
        # Pick 3 representative epochs: 0, stage1, stage2
        key_epochs = [0]
        ep_s1 = min(epochs, key=lambda e: abs(e - stage1_end))
        ep_s2 = min(epochs, key=lambda e: abs(e - stage2_end))
        if ep_s1 not in key_epochs:
            key_epochs.append(ep_s1)
        if ep_s2 not in key_epochs:
            key_epochs.append(ep_s2)
        while len(key_epochs) < 3:
            key_epochs.append(epochs[-1])

        num_components = min(20, d_mlp)
        n = len(key_epochs)
        fig, axes = plt.subplots(
            2, n, figsize=(18, 3.3 * n),
            gridspec_kw={"hspace": 0.05}, constrained_layout=True
        )
        if n == 1:
            axes = axes.reshape(2, 1)

        x_locs = np.arange(len(self.fourier_basis_names))
        y_locs = np.arange(num_components)

        for col, key in enumerate(key_epochs):
            W_in = checkpoints[key]['mlp.W_in']
            W_out = checkpoints[key]['mlp.W_out']

            data_in = (W_in @ self.fourier_basis.T)[:num_components]
            data_in_np = data_in.detach().cpu().numpy()
            abs_max_in = np.abs(data_in_np).max()
            ax_in = axes[0, col]
            im_in = ax_in.imshow(
                data_in_np, cmap=CMAP_DIVERGING,
                vmin=-abs_max_in, vmax=abs_max_in, aspect='auto'
            )
            ax_in.set_title(rf'Step {key}, $\theta_m$ after DFT', fontsize=18)
            ax_in.set_xticks(x_locs)
            ax_in.set_xticklabels(self.fourier_basis_names, rotation=90, fontsize=11)
            ax_in.set_yticks(y_locs)
            ax_in.set_yticklabels(y_locs)
            if col == 0:
                ax_in.set_ylabel('Neuron #', fontsize=16)
            fig.colorbar(im_in, ax=ax_in)

            data_out = (W_out.T @ self.fourier_basis.T)[:num_components]
            data_out_np = data_out.detach().cpu().numpy()
            abs_max_out = np.abs(data_out_np).max() * 0.85
            ax_out = axes[1, col]
            im_out = ax_out.imshow(
                data_out_np, cmap=CMAP_DIVERGING,
                vmin=-abs_max_out, vmax=abs_max_out, aspect='auto'
            )
            ax_out.set_title(rf'Step {key}, $\xi_m$ after DFT', fontsize=18)
            ax_out.set_xticks(x_locs)
            ax_out.set_xticklabels(self.fourier_basis_names, rotation=90, fontsize=11)
            ax_out.set_yticks(y_locs)
            ax_out.set_yticklabels(y_locs)
            if col == 0:
                ax_out.set_ylabel('Neuron #', fontsize=16)
            fig.colorbar(im_out, ax=ax_out)

        _save_fig(fig, self._out('grokk_decoded_weights_dynamic.png'))

        print("    Saved grokk_loss.json, grokk_loss.png, grokk_acc.json, grokk_acc.png, "
              "grokk_abs_phase_diff.png, grokk_avg_ipr.png, "
              "grokk_memorization_accuracy.png, "
              "grokk_memorization_common_to_rare.png, grokk_decoded_weights_dynamic.png")

    # ------------------------------------------------------------------
    # Tab 6: Lottery Mechanism
    # ------------------------------------------------------------------

    def generate_tab6(self):
        """Generate lottery mechanism plots."""
        print(f"  [Tab 6] Lottery Mechanism for p={self.p}")
        run_dir = self._run_dir('quad_random')
        if run_dir is None:
            print("    SKIP: quad_random run directory not found")
            return

        checkpoints = _load_checkpoints(run_dir, self.device)
        if not checkpoints:
            print("    SKIP: no quad_random checkpoints")
            return

        final_data = _load_final(run_dir, self.device)
        if final_data is None:
            print("    SKIP: no final quad_random checkpoint")
            return
        model_load_final = final_data['model']

        # Select best neuron
        neuron_id = select_lottery_neuron(
            model_load_final, self.fourier_basis, decode_scales_phis
        )

        epochs = sorted(checkpoints.keys())
        p = self.p

        # Collect per-checkpoint scales and phase diffs for the selected neuron
        scales_list = []
        diff_list = []
        for ep in epochs:
            scales, phis, psis = decode_scales_phis(
                checkpoints[ep], self.fourier_basis
            )
            scales_list.append(scales[neuron_id])
            diff_list.append(normalize_to_pi(
                psis[neuron_id] - 2 * phis[neuron_id]
            ))

        # Stack: [num_checkpoints, K+1], skip DC
        scales_all = torch.stack(scales_list, dim=0)[:, 1:]
        diff_all = torch.stack(diff_list, dim=0)[:, 1:]

        # Determine which frequency this neuron specializes in
        _, _, max_freq_ls = decode_weights(model_load_final, self.fourier_basis)
        max_freq = max_freq_ls[neuron_id] - 1  # 0-indexed into scales_all

        scales_np = scales_all.cpu().numpy()
        diff_np = diff_all.cpu().numpy()
        num_models, num_freqs = scales_np.shape
        n_plot = min(num_models, 160)
        scales_np = scales_np[:n_plot]
        diff_np = diff_np[:n_plot]
        x_idx = np.arange(n_plot)

        # Color gradient for non-highlighted frequencies
        base_rgb = np.array(mcolors.to_rgb(COLORS[0]))
        gray_rgb = np.array(mcolors.to_rgb('white'))
        highlight_color = COLORS[3]

        nonmax = [f for f in range(num_freqs) if f != max_freq]
        final_scales = scales_np[-1]
        sorted_nonmax = sorted(nonmax, key=lambda f: final_scales[f])
        M = len(sorted_nonmax)

        # Compute save_every for x-axis formatter
        save_every = epochs[1] - epochs[0] if len(epochs) > 1 else 200

        # ---- Magnitude plot ----
        fig, ax = plt.subplots(figsize=(4, 4))
        for idx, f in enumerate(sorted_nonmax):
            blend = idx / (M - 1) if M > 1 else 0.0
            col_rgb = (1 - blend - 0.05) * gray_rgb + (blend + 0.05) * base_rgb
            ax.plot(x_idx, scales_np[:, f], color=col_rgb, linestyle=':',
                    marker='x', linewidth=3.5, markersize=1.5,
                    label=f"Freq. {f + 1}")

        ax.plot(x_idx, scales_np[:, max_freq], color=highlight_color,
                linestyle=':', marker='x', linewidth=3.5, markersize=1.5,
                label=f"Freq. {max_freq + 1}")

        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda val, pos: f"{int(val * save_every)}")
        )
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                  borderaxespad=0.2, frameon=False, fontsize=13)
        ax.set_xlabel("Step", fontsize=16)
        ax.set_ylabel("Magnitude", fontsize=16)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        _save_fig(fig, self._out('lottery_mech_magnitude.png'))

        # ---- Phase misalignment plot ----
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axhline(y=0, color='black', linewidth=1, linestyle='dotted')
        for idx, f in enumerate(sorted_nonmax):
            blend = idx / (M - 1) if M > 1 else 0.0
            col_rgb = (1 - blend - 0.05) * gray_rgb + (blend + 0.05) * base_rgb
            ax.plot(x_idx, diff_np[:, f], linestyle=':', marker='x',
                    linewidth=3.5, markersize=1.5, color=col_rgb,
                    label=f"Freq. {f}")

        ax.plot(x_idx, diff_np[:, max_freq], linestyle=':', marker='x',
                linewidth=3.5, markersize=1.5, color=highlight_color,
                label=f"Freq. {max_freq}")

        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda val, pos: f"{int(val * save_every)}")
        )
        ax.set_xlabel("Step", fontsize=16)
        ax.set_ylabel("Misalignment", fontsize=16)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        _save_fig(fig, self._out('lottery_mech_phase.png'))

        # ---- Beta contour: simulate gradient flow ----
        self._generate_lottery_contour()

        print("    Saved lottery_mech_magnitude.png, lottery_mech_phase.png, "
              "lottery_beta_contour.png")

    def _generate_lottery_contour(self):
        """Simulate gradient flow for a grid of (init_magnitude, init_phase_diff)."""
        p = self.p
        device = self.device
        init_k = 1
        init_psi = 0.0
        num_steps = 100
        learning_rate = 0.01

        fourier_basis, _ = get_fourier_basis(p, device)
        fourier_basis = fourier_basis.to(torch.get_default_dtype())

        initial_scales = np.linspace(0.01, 0.02, num=30)
        phi0_vals = np.linspace(0, np.pi, num=30)

        results = []
        for scale in initial_scales:
            for phi0 in phi0_vals:
                w_k = 2 * np.pi * init_k / p
                theta = scale * torch.tensor(
                    [np.cos(w_k * j + phi0) for j in range(p)],
                    device=device
                )
                xi = scale * torch.tensor(
                    [np.cos(w_k * j + init_psi) for j in range(p)],
                    device=device
                )

                # Run gradient flow simulation
                for _ in range(num_steps):
                    theta, xi = self._gradient_flow_step(
                        theta, xi, init_k, p, learning_rate, fourier_basis
                    )

                # Compute final beta
                coeffs_xi = fourier_basis.to(xi.dtype) @ xi
                idx = [init_k * 2 - 1, init_k * 2]
                xi_n = coeffs_xi[idx]
                beta_f = torch.norm(xi_n).item() * np.sqrt(2 / p)

                results.append({
                    "init_scale": scale,
                    "init_diff": 2 * phi0,
                    "beta_f": beta_f,
                })

        # Pivot into grid
        n_scales = len(initial_scales)
        n_phis = len(phi0_vals)
        Z = np.zeros((n_phis, n_scales))
        for i, r in enumerate(results):
            row = i % n_phis
            col = i // n_phis
            Z[row, col] = r['beta_f']

        X, Y = np.meshgrid(initial_scales, 2 * phi0_vals)

        fig = plt.figure(figsize=(4.5, 4))
        cf = plt.contourf(X, Y, Z, levels=12, cmap=CMAP_DIVERGING, extend='both')
        plt.axhline(y=np.pi, color='white', linewidth=1, linestyle=':')
        plt.xlabel("Initial Magnitude", fontsize=16)
        plt.ylabel("Initial Phase Difference", fontsize=16)
        plt.title("Contour of Final Magnitude", fontsize=16)
        plt.colorbar(cf)
        plt.tight_layout()
        _save_fig(fig, self._out('lottery_beta_contour.png'))

    @staticmethod
    def _gradient_flow_step(theta, xi, init_k, p, lr, fourier_basis):
        """One step of analytical gradient flow."""
        fb = fourier_basis.to(theta.dtype)
        theta_coeff = fb @ theta
        xi_coeff = fb @ xi

        neuron_coeff_theta = theta_coeff[[init_k * 2 - 1, init_k * 2]]
        alpha = np.sqrt(2 / p) * torch.sqrt(
            torch.sum(neuron_coeff_theta.pow(2))
        ).item()
        phi = np.arctan2(
            -neuron_coeff_theta[1].item(), neuron_coeff_theta[0].item()
        )

        neuron_coeff_xi = xi_coeff[[init_k * 2 - 1, init_k * 2]]
        beta = np.sqrt(2 / p) * torch.sqrt(
            torch.sum(neuron_coeff_xi.pow(2))
        ).item()
        psi = np.arctan2(
            -neuron_coeff_xi[1].item(), neuron_coeff_xi[0].item()
        )

        w_k = 2 * np.pi * init_k / p
        grad_theta = torch.tensor(
            [2 * p * alpha * beta * np.cos(w_k * j + psi - phi)
             for j in range(p)],
            device=theta.device
        )
        grad_xi = torch.tensor(
            [p * alpha ** 2 * np.cos(w_k * j + 2 * phi)
             for j in range(p)],
            device=theta.device
        )

        theta = theta + lr * grad_theta
        xi = xi + lr * grad_xi
        return theta, xi

    # ------------------------------------------------------------------
    # Tab 7: Gradient Dynamics
    # ------------------------------------------------------------------

    def generate_tab7(self):
        """Generate gradient dynamics plots for quad_single_freq and relu_single_freq."""
        print(f"  [Tab 7] Gradient Dynamics for p={self.p}")

        for run_name, act_name, prefix in [
            ('quad_single_freq', 'Quad', 'quad'),
            ('relu_single_freq', 'ReLU', 'relu'),
        ]:
            run_dir = self._run_dir(run_name)
            if run_dir is None:
                print(f"    SKIP: {run_name} run directory not found")
                continue

            checkpoints = _load_checkpoints(run_dir, self.device)
            if not checkpoints:
                print(f"    SKIP: no {run_name} checkpoints")
                continue

            epochs = sorted(checkpoints.keys())
            d_mlp = self.d_mlp

            # Build all neuron records across epochs
            all_neuron_records = []
            for ep in epochs:
                model_sd = checkpoints[ep]
                W_in_d, W_out_d, mfl = decode_weights(model_sd, self.fourier_basis)
                for neuron in range(W_in_d.shape[0]):
                    s_in, phi_in = compute_neuron(neuron, mfl, W_in_d)
                    s_out, phi_out = compute_neuron(neuron, mfl, W_out_d)
                    all_neuron_records.append({
                        'epoch': ep,
                        'neuron': neuron,
                        'scale_in': s_in,
                        'phi_in': phi_in,
                        'scale_out': s_out,
                        'phi_out': phi_out,
                    })

            # Select a neuron that shows interesting phase convergence dynamics.
            # The lottery winner (largest final scale) already has ψ ≈ 2φ from
            # the start, producing flat boring plots. Instead, pick a neuron that
            # (a) has significant final scale (top quartile → actually learned),
            # (b) had the largest initial phase misalignment |ψ₀ - 2φ₀|.
            final_records = [r for r in all_neuron_records if r['epoch'] == epochs[-1]]
            if not final_records:
                continue
            init_records = [r for r in all_neuron_records if r['epoch'] == epochs[0]]
            init_by_neuron = {r['neuron']: r for r in init_records}

            # Keep neurons with final scale in top 25%
            scales = sorted([r['scale_in'] for r in final_records], reverse=True)
            scale_threshold = scales[max(0, len(scales) // 4 - 1)] if len(scales) >= 4 else scales[-1]
            strong_neurons = [r for r in final_records if r['scale_in'] >= scale_threshold]

            # Among strong neurons, pick the one with largest initial misalignment
            best_neuron = None
            best_misalign = -1.0
            for r in strong_neurons:
                n = r['neuron']
                if n not in init_by_neuron:
                    continue
                ir = init_by_neuron[n]
                misalign = abs(normalize_to_pi(ir['phi_out'] - 2 * ir['phi_in']))
                if misalign > best_misalign:
                    best_misalign = misalign
                    best_neuron = n
            if best_neuron is None:
                best_neuron = max(final_records, key=lambda r: r['scale_in'])['neuron']

            # Extract trajectory for this neuron
            neuron_records = [r for r in all_neuron_records if r['neuron'] == best_neuron]
            # Remove last few points if noisy (as notebooks do)
            trim = max(0, len(neuron_records) - 4) if prefix == 'relu' else max(0, len(neuron_records) - 14)
            neuron_records = neuron_records[:trim] if trim > 0 else neuron_records

            phi_in_raw = [r['phi_in'] for r in neuron_records]
            phi_out_raw = [r['phi_out'] for r in neuron_records]
            scale_in_list = [r['scale_in'] for r in neuron_records]
            scale_out_list = [r['scale_out'] for r in neuron_records]

            # Phase wrapping fix: normalize 2*phi to [-pi, pi], then adjust
            # psi to stay within pi of 2*phi (same fix as Tab 3 scatter).
            phi2_in_list = [normalize_to_pi(2 * v) for v in phi_in_raw]
            phi_out_list = []
            for two_phi, psi in zip(phi2_in_list, phi_out_raw):
                psi_n = normalize_to_pi(psi)
                if psi_n - two_phi > np.pi:
                    psi_n -= 2 * np.pi
                elif psi_n - two_phi < -np.pi:
                    psi_n += 2 * np.pi
                phi_out_list.append(psi_n)

            # Unwrap time series to remove remaining jumps at +-pi boundary
            phi_in_list = list(np.unwrap(phi_in_raw))
            phi2_in_list = list(np.unwrap(phi2_in_list))
            phi_out_list = list(np.unwrap(phi_out_list))

            x = np.arange(len(phi_in_list)) * (epochs[1] - epochs[0] if len(epochs) > 1 else 200)

            # ---- Phase alignment + magnitude plot ----
            fig_width = 8 if prefix == 'quad' else 5
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 6), sharex=True)

            ax1.plot(x, phi_in_list, marker='o', markersize=4,
                     color=COLORS[1], label=r"$\phi_m^\star$")
            ax1.plot(x, phi_out_list, marker='x', markersize=4,
                     color=COLORS[3], label=r"$\psi_m^\star$")
            ax1.plot(x, phi2_in_list, marker='^', markersize=4,
                     color=COLORS[0], label=r"$2\phi_m^\star$")
            ax1.set_title('Phase Alignment of Neuron $m$', fontsize=16)
            ax1.legend(fontsize=18, loc="upper right")
            ax1.grid(True)

            ax2.plot(x, scale_in_list, marker='o', markersize=4,
                     color=COLORS[0], label=r"$\alpha_m^\star$")
            ax2.plot(x, scale_out_list, marker='x', markersize=4,
                     color=COLORS[3], label=r"$\beta_m^\star$")
            ax2.set_title('Magnitude Growth of Neuron $m$', fontsize=16)
            ax2.set_xlabel('Step', fontsize=16)
            ax2.legend(fontsize=18, loc="upper left")
            ax2.grid(True)

            plt.tight_layout()
            _save_fig(fig, self._out(f'phase_align_{prefix}.png'))

            # ---- Decoded weights at timepoints ----
            # Use fixed timepoints: steps 0, 1000, 5000 for both Quad and ReLU
            target_keys = [0, 1000, 5000]
            # Snap each target to the nearest available checkpoint epoch
            keys = []
            for t in target_keys:
                nearest = min(epochs, key=lambda e: abs(e - t))
                if nearest not in keys:
                    keys.append(nearest)

            num_components = min(20, d_mlp)
            n = len(keys)
            fig, axes = plt.subplots(
                2, n, figsize=(12 if n <= 2 else 18, 4 * n if n <= 2 else 3.3 * n),
                gridspec_kw={"hspace": 0.05}, constrained_layout=True
            )
            if n == 1:
                axes = axes.reshape(2, 1)

            x_locs = np.arange(len(self.fourier_basis_names))
            y_locs = np.arange(num_components)

            for col, key in enumerate(keys):
                if key not in checkpoints:
                    key = min(checkpoints.keys(), key=lambda e: abs(e - key))
                W_in = checkpoints[key]['mlp.W_in']
                W_out = checkpoints[key]['mlp.W_out']

                data_in = (W_in @ self.fourier_basis.T)[:num_components]
                data_in_np = data_in.detach().cpu().numpy()
                abs_max_in = np.abs(data_in_np).max()
                ax_in = axes[0, col]
                im_in = ax_in.imshow(
                    data_in_np, cmap=CMAP_DIVERGING,
                    vmin=-abs_max_in, vmax=abs_max_in, aspect='auto'
                )
                ax_in.set_title(rf'Step {key}, $\theta_m$ after DFT', fontsize=18)
                ax_in.set_xticks(x_locs)
                ax_in.set_xticklabels(self.fourier_basis_names, rotation=90, fontsize=11)
                ax_in.set_yticks(y_locs)
                ax_in.set_yticklabels(y_locs)
                if col == 0:
                    ax_in.set_ylabel('Neuron #', fontsize=16)
                fig.colorbar(im_in, ax=ax_in)

                data_out = (W_out.T @ self.fourier_basis.T)[:num_components]
                data_out_np = data_out.detach().cpu().numpy()
                abs_max_out = np.abs(data_out_np).max()
                ax_out = axes[1, col]
                im_out = ax_out.imshow(
                    data_out_np, cmap=CMAP_DIVERGING,
                    vmin=-abs_max_out, vmax=abs_max_out, aspect='auto'
                )
                ax_out.set_title(rf'Step {key}, $\xi_m$ after DFT', fontsize=18)
                ax_out.set_xticks(x_locs)
                ax_out.set_xticklabels(self.fourier_basis_names, rotation=90, fontsize=11)
                ax_out.set_yticks(y_locs)
                ax_out.set_yticklabels(y_locs)
                if col == 0:
                    ax_out.set_ylabel('Neuron #', fontsize=16)
                fig.colorbar(im_out, ax=ax_out)

            _save_fig(fig, self._out(f'single_freq_{prefix}.png'))

            print(f"    Saved phase_align_{prefix}.png, single_freq_{prefix}.png")

    # ------------------------------------------------------------------
    # Metadata JSON
    # ------------------------------------------------------------------

    def _save_metadata(self):
        """Save a metadata JSON summarizing config and final metrics."""
        print(f"  [Meta] Saving metadata for p={self.p}")
        meta = {
            'prime': self.p,
            'd_mlp': self.d_mlp,
            'training_runs': {},
            'final_metrics': {},
        }
        for run_name, params in TRAINING_RUNS.items():
            meta['training_runs'][run_name] = {
                'act_type': params['act_type'],
                'lr': params['lr'],
                'weight_decay': params['weight_decay'],
                'num_epochs': params['num_epochs'],
                'frac_train': params['frac_train'],
                'init_type': params['init_type'],
                'init_scale': params['init_scale'],
                'optimizer': params['optimizer'],
            }
            curves = _load_training_curves(self._run_type_dir(run_name))
            if curves:
                metrics = {}
                if 'train_accs' in curves and curves['train_accs']:
                    metrics['train_acc'] = curves['train_accs'][-1]
                if 'test_accs' in curves and curves['test_accs']:
                    metrics['test_acc'] = curves['test_accs'][-1]
                if 'train_losses' in curves and curves['train_losses']:
                    metrics['train_loss'] = curves['train_losses'][-1]
                if 'test_losses' in curves and curves['test_losses']:
                    metrics['test_loss'] = curves['test_losses'][-1]
                if metrics:
                    meta['final_metrics'][run_name] = metrics

        with open(self._out('metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print("    Saved metadata.json")

    # ------------------------------------------------------------------
    # Interactive JSON precomputation
    # ------------------------------------------------------------------

    def _precompute_neuron_spectra(self):
        """Precompute per-neuron Fourier magnitude spectra for top-20 neurons."""
        print(f"  [Interactive] Neuron spectra for p={self.p}")
        run_dir = self._run_dir('standard')
        if run_dir is None:
            print("    SKIP: standard run directory not found")
            return

        final_data = _load_final(run_dir, self.device)
        if final_data is None:
            print("    SKIP: no final checkpoint")
            return
        model_load = final_data['model']

        W_in_decode, W_out_decode, max_freq_ls = decode_weights(
            model_load, self.fourier_basis
        )
        d_mlp = W_in_decode.shape[0]
        num_neurons = min(20, d_mlp)

        sorted_indices = select_top_neurons_by_frequency(
            max_freq_ls, W_in_decode, n=num_neurons
        )

        fb_names = self.fourier_basis_names
        spectra = {}
        for rank, neuron_idx in enumerate(sorted_indices):
            # Fourier magnitudes for W_in
            magnitudes_in = W_in_decode[neuron_idx].abs().cpu().tolist()
            magnitudes_out = W_out_decode[neuron_idx].abs().cpu().tolist()
            spectra[f"neuron_{rank}"] = {
                'global_index': int(neuron_idx),
                'dominant_freq': int(max_freq_ls[neuron_idx]),
                'fourier_magnitudes_in': magnitudes_in,
                'fourier_magnitudes_out': magnitudes_out,
            }

        payload = {
            'fourier_basis_names': fb_names,
            'neurons': spectra,
        }
        with open(self._out('neuron_spectra.json'), 'w') as f:
            json.dump(payload, f)
        print("    Saved neuron_spectra.json")

    def _precompute_logit_explorer(self):
        """Precompute logits for representative (a,b) pairs."""
        print(f"  [Interactive] Logit explorer for p={self.p}")
        run_dir = self._run_dir('standard')
        if run_dir is None:
            print("    SKIP: standard run directory not found")
            return

        final_data = _load_final(run_dir, self.device)
        if final_data is None:
            print("    SKIP: no final checkpoint")
            return
        model_load = final_data['model']

        p = self.p
        act_type = TRAINING_RUNS['standard']['act_type']
        model = EmbedMLP(
            d_vocab=self.d_vocab, d_model=self.d_model,
            d_mlp=self.d_mlp, act_type=act_type, use_cache=False
        )
        model.to(self.device)
        model.load_state_dict(model_load)
        model.eval()

        # Select p representative pairs: (0,0), (1,2), (3,5), ... spread across inputs
        pairs = []
        step = max(1, (p * p) // p)
        for idx in range(0, p * p, step):
            a = idx // p
            b = idx % p
            pairs.append((a, b))
            if len(pairs) >= p:
                break

        pair_tensor = torch.tensor(pairs, dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = model(pair_tensor).squeeze(1)  # [n_pairs, p]

        payload = {
            'pairs': pairs,
            'correct_answers': [(a + b) % p for a, b in pairs],
            'logits': logits.cpu().tolist(),
            'output_classes': list(range(p)),
        }
        with open(self._out('logits_interactive.json'), 'w') as f:
            json.dump(payload, f)
        print("    Saved logits_interactive.json")

    def _precompute_grokk_slider(self):
        """Precompute accuracy grids at ~10 grokking checkpoints for epoch slider."""
        print(f"  [Interactive] Grokking epoch slider for p={self.p}")
        if self.p < MIN_P_GROKKING:
            print(f"    SKIP: p={self.p} < {MIN_P_GROKKING}")
            return
        run_dir = self._run_dir('grokking')
        if run_dir is None:
            print("    SKIP: grokking run directory not found")
            return

        checkpoints = _load_checkpoints(run_dir, self.device)
        if not checkpoints:
            print("    SKIP: no grokking checkpoints")
            return

        epochs = sorted(checkpoints.keys())
        p = self.p
        d_mlp = self.d_mlp
        act_type = TRAINING_RUNS['grokking']['act_type']
        gt_grid = self.all_labels.view(p, p)

        # Subsample ~10 epochs evenly
        n_snapshots = min(10, len(epochs))
        indices = np.linspace(0, len(epochs) - 1, n_snapshots, dtype=int)
        selected_epochs = [epochs[i] for i in indices]

        epoch_data = []
        for ep in selected_epochs:
            model = EmbedMLP(
                d_vocab=self.d_vocab, d_model=self.d_model,
                d_mlp=d_mlp, act_type=act_type, use_cache=False
            ).to(self.device)
            model.load_state_dict(checkpoints[ep])
            model.eval()
            with torch.no_grad():
                logits = model(self.all_data).squeeze(1)
            predicted = torch.argmax(logits, dim=1).view(p, p)
            accuracy_grid = (predicted == gt_grid).float().cpu().tolist()
            epoch_data.append({
                'epoch': int(ep),
                'accuracy_grid': accuracy_grid,
            })

        payload = {
            'prime': p,
            'epochs': [d['epoch'] for d in epoch_data],
            'grids': [d['accuracy_grid'] for d in epoch_data],
        }
        with open(self._out('grokk_epoch_data.json'), 'w') as f:
            json.dump(payload, f)
        print("    Saved grokk_epoch_data.json")

    # ------------------------------------------------------------------
    # Training Log consolidation
    # ------------------------------------------------------------------

    def _save_training_log(self):
        """Consolidate training logs from all runs into a precomputed JSON.

        For each run, includes:
        - config: hyperparameters
        - log_text: human-readable formatted log
        - table: subsampled per-epoch metrics for display
        """
        print(f"  [Log] Saving training log for p={self.p}")
        all_runs = {}

        for run_name, params in TRAINING_RUNS.items():
            run_type_dir = self._run_type_dir(run_name)
            curves = _load_training_curves(run_type_dir)
            if curves is None:
                continue

            # Also check for a pre-saved training_log.txt
            log_text_path = os.path.join(run_type_dir, "training_log.txt")
            if os.path.exists(log_text_path):
                with open(log_text_path) as f:
                    log_text = f.read()
            else:
                # Reconstruct from curves data
                log_text = self._reconstruct_log_text(
                    run_name, params, curves
                )

            # Build a subsampled table (~100 rows max)
            n_epochs = len(curves.get('train_losses', []))
            step = max(1, n_epochs // 100)
            indices = list(range(0, n_epochs, step))
            if n_epochs > 0 and (n_epochs - 1) not in indices:
                indices.append(n_epochs - 1)

            table = []
            for i in indices:
                row = {'epoch': i}
                for key in ('train_losses', 'test_losses', 'train_accs',
                            'test_accs', 'grad_norms', 'param_norms'):
                    vals = curves.get(key, [])
                    row[key.replace('_', '_')] = (
                        round(vals[i], 6) if i < len(vals) else None
                    )
                table.append(row)

            all_runs[run_name] = {
                'config': {
                    'prime': self.p,
                    'd_mlp': self.d_mlp,
                    'act_type': params['act_type'],
                    'init_type': params['init_type'],
                    'init_scale': params['init_scale'],
                    'optimizer': params['optimizer'],
                    'lr': params['lr'],
                    'weight_decay': params['weight_decay'],
                    'frac_train': params['frac_train'],
                    'num_epochs': params['num_epochs'],
                    'seed': params['seed'],
                },
                'log_text': log_text,
                'table': table,
                'total_epochs': n_epochs,
            }

        if all_runs:
            with open(self._out('training_log.json'), 'w') as f:
                json.dump(all_runs, f)
            print(f"    Saved training_log.json ({len(all_runs)} runs)")
        else:
            print("    SKIP: no training curves found")

    def _reconstruct_log_text(self, run_name, params, curves):
        """Reconstruct a human-readable training log from curves data."""
        lines = []
        lines.append(f"{'=' * 70}")
        lines.append(f"Training Log: p={self.p}, run={run_name}")
        lines.append(f"{'=' * 70}")
        lines.append("")
        lines.append("Configuration:")
        lines.append(f"  prime (p)       = {self.p}")
        lines.append(f"  d_mlp           = {self.d_mlp}")
        lines.append(f"  activation      = {params['act_type']}")
        lines.append(f"  init_type       = {params['init_type']}")
        lines.append(f"  init_scale      = {params['init_scale']}")
        lines.append(f"  optimizer       = {params['optimizer']}")
        lines.append(f"  learning_rate   = {params['lr']}")
        lines.append(f"  weight_decay    = {params['weight_decay']}")
        lines.append(f"  frac_train      = {params['frac_train']}")
        lines.append(f"  num_epochs      = {params['num_epochs']}")
        lines.append(f"  seed            = {params['seed']}")
        lines.append("")
        lines.append(f"{'─' * 70}")
        lines.append(
            f"{'Epoch':>8s}  {'Train Loss':>12s}  {'Test Loss':>12s}  "
            f"{'Train Acc':>10s}  {'Test Acc':>10s}  "
            f"{'Grad Norm':>10s}  {'Param Norm':>11s}"
        )
        lines.append(f"{'─' * 70}")

        train_losses = curves.get('train_losses', [])
        test_losses = curves.get('test_losses', [])
        train_accs = curves.get('train_accs', [])
        test_accs = curves.get('test_accs', [])
        grad_norms = curves.get('grad_norms', [])
        param_norms = curves.get('param_norms', [])
        n_epochs = len(train_losses)

        step = max(1, n_epochs // 100)
        indices = list(range(0, n_epochs, step))
        if n_epochs > 0 and (n_epochs - 1) not in indices:
            indices.append(n_epochs - 1)

        for i in indices:
            tl = f"{train_losses[i]:.6f}" if i < len(train_losses) else "N/A"
            tel = f"{test_losses[i]:.6f}" if i < len(test_losses) else "N/A"
            ta = f"{train_accs[i]:.4f}" if i < len(train_accs) else "N/A"
            tea = f"{test_accs[i]:.4f}" if i < len(test_accs) else "N/A"
            gn = f"{grad_norms[i]:.4f}" if i < len(grad_norms) else "N/A"
            pn = f"{param_norms[i]:.4f}" if i < len(param_norms) else "N/A"
            lines.append(
                f"{i:>8d}  {tl:>12s}  {tel:>12s}  "
                f"{ta:>10s}  {tea:>10s}  "
                f"{gn:>10s}  {pn:>11s}"
            )

        lines.append(f"{'─' * 70}")
        lines.append("")
        lines.append("Final Results:")
        if train_losses:
            lines.append(f"  Train Loss  = {train_losses[-1]:.6f}")
        if test_losses:
            lines.append(f"  Test Loss   = {test_losses[-1]:.6f}")
        if train_accs:
            lines.append(f"  Train Acc   = {train_accs[-1]:.4f}")
        if test_accs:
            lines.append(f"  Test Acc    = {test_accs[-1]:.4f}")
        if param_norms:
            lines.append(f"  Param Norm  = {param_norms[-1]:.4f}")
        lines.append(f"\nTotal epochs trained: {n_epochs}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Generate all
    # ------------------------------------------------------------------

    def generate_all(self):
        """Generate all tab plots with error handling."""
        print(f"\n{'=' * 60}")
        print(f"Generating plots for p={self.p}")
        print(f"  Input:  {self.input_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"{'=' * 60}")

        # Save metadata and training logs first
        try:
            self._save_metadata()
        except Exception as e:
            print(f"  [ERROR] metadata failed: {e}")
            traceback.print_exc()

        try:
            self._save_training_log()
        except Exception as e:
            print(f"  [ERROR] training log failed: {e}")
            traceback.print_exc()

        generators = [
            ('Tab 1', self.generate_tab1),
            ('Tab 2', self.generate_tab2),
            ('Tab 3', self.generate_tab3),
            ('Tab 4', self.generate_tab4),
            ('Tab 5', self.generate_tab5),
            ('Tab 6', self.generate_tab6),
            ('Tab 7', self.generate_tab7),
        ]

        for name, gen_fn in generators:
            try:
                gen_fn()
            except Exception as e:
                print(f"  [ERROR] {name} failed: {e}")
                traceback.print_exc()

        # Precompute interactive JSON data
        interactive = [
            ('Neuron Spectra', self._precompute_neuron_spectra),
            ('Logit Explorer', self._precompute_logit_explorer),
            ('Grokking Slider', self._precompute_grokk_slider),
        ]
        for name, fn in interactive:
            try:
                fn()
            except Exception as e:
                print(f"  [ERROR] {name} failed: {e}")
                traceback.print_exc()

        print(f"\nDone generating plots for p={self.p}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate all model-dependent plots for the HF app.'
    )
    parser.add_argument('--all', action='store_true',
                        help='Generate plots for all p found in input dir')
    parser.add_argument('--p', type=int,
                        help='Generate plots for a specific p')
    parser.add_argument('--input', type=str, default='./trained_models',
                        help='Base input directory containing p_PPP subdirs')
    parser.add_argument('--output', type=str,
                        default='./precomputed_results',
                        help='Base output directory for precomputed results')
    args = parser.parse_args()

    if not args.all and args.p is None:
        parser.error("Specify --all or --p P")

    if args.p:
        moduli = [args.p]
    else:
        # Discover moduli from input directory
        moduli = []
        if os.path.isdir(args.input):
            for d in sorted(os.listdir(args.input)):
                if d.startswith('p_'):
                    try:
                        p = int(d.split('_')[1])
                        moduli.append(p)
                    except (ValueError, IndexError):
                        pass
        if not moduli:
            print(f"No p_PPP directories found in {args.input}")
            sys.exit(1)

    total = len(moduli)
    for i, p in enumerate(moduli):
        print(f"\n[{i + 1}/{total}] Processing p={p}")
        # Handle both p_23 and p_023 naming conventions
        input_dir = os.path.join(args.input, f'p_{p:03d}')
        if not os.path.isdir(input_dir):
            input_dir = os.path.join(args.input, f'p_{p}')
        if not os.path.isdir(input_dir):
            print(f"  Input directory not found: {input_dir}")
            continue

        output_dir = os.path.join(args.output, f'p_{p:03d}')

        gen = PlotGenerator(p=p, input_dir=input_dir, output_dir=output_dir)
        gen.generate_all()

    print(f"\nAll done. Processed {total} prime(s).")


if __name__ == '__main__':
    main()
