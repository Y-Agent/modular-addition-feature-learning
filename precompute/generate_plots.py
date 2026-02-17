#!/usr/bin/env python3
"""
Main plot generation script for the HF app.
Creates all model-dependent plots (Tabs 1-7) from trained checkpoints.

Usage:
    python generate_plots.py --all               # Generate for all primes
    python generate_plots.py --prime 23           # Generate for a specific prime
    python generate_plots.py --prime 23 --input ./trained_models --output ./hf_app/precomputed_results
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
from precompute.prime_config import compute_d_mlp, TRAINING_RUNS

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
        self.d_mlp = compute_d_mlp(p)
        self.d_vocab = p
        self.d_model = p

        os.makedirs(output_dir, exist_ok=True)

        # Fourier basis (mechanism_base version with device arg)
        self.fourier_basis, self.fourier_basis_names = get_fourier_basis(p, self.device)

        # All (a,b) pairs and labels
        self.all_data = torch.tensor(
            [(i, j) for i in range(p) for j in range(p)], dtype=torch.long
        )
        self.all_labels = torch.tensor(
            [(i + j) % p for i in range(p) for j in range(p)], dtype=torch.long
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _run_type_dir(self, run_name):
        return os.path.join(self.input_dir, run_name)

    def _run_dir(self, run_name):
        return _find_run_dir(self._run_type_dir(run_name))

    def _out(self, filename):
        return os.path.join(self.output_dir, filename)

    # ------------------------------------------------------------------
    # Tab 1: Training Overview (loss + IPR over epochs)
    # ------------------------------------------------------------------

    def generate_tab1(self):
        """Generate loss_sparsity.png and loss_sparsity.json."""
        print(f"  [Tab 1] Training Overview for p={self.p}")
        run_dir = self._run_dir('standard')
        if run_dir is None:
            print("    SKIP: standard run directory not found")
            return

        curves = _load_training_curves(self._run_type_dir('standard'))
        checkpoints = _load_checkpoints(run_dir, self.device)

        if not checkpoints:
            print("    SKIP: no checkpoints found")
            return

        epochs_sorted = sorted(checkpoints.keys())

        # Compute IPR at each checkpoint
        ipr_values = []
        for ep in epochs_sorted:
            model_sd = checkpoints[ep]
            W_in_decode, W_out_decode, _ = decode_weights(model_sd, self.fourier_basis)
            ipr_in = (W_in_decode.norm(p=4, dim=1) ** 4
                      / W_in_decode.norm(p=2, dim=1) ** 4).mean() / 2
            ipr_out = (W_out_decode.norm(p=4, dim=1) ** 4
                       / W_out_decode.norm(p=2, dim=1) ** 4).mean() / 2
            ipr_values.append((ipr_in + ipr_out).item())

        # Get loss values: prefer curves JSON, otherwise subsample from checkpoints
        if curves and 'train_losses' in curves:
            save_every = epochs_sorted[1] - epochs_sorted[0] if len(epochs_sorted) > 1 else 200
            loss_values = curves['train_losses'][::save_every]
            # Align lengths
            min_len = min(len(loss_values), len(ipr_values))
            loss_values = loss_values[:min_len]
            ipr_values_plot = ipr_values[:min_len]
            x = np.array(epochs_sorted[:min_len])
        else:
            loss_values = None
            ipr_values_plot = ipr_values
            x = np.array(epochs_sorted)

        # --- Plot ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

        if loss_values is not None:
            ax1.plot(x, loss_values, marker='o', markersize=4,
                     color=COLORS[0], label="Loss")
        ax1.set_title('Training Loss', fontsize=16)
        ax1.legend(fontsize=18, loc="upper right")
        ax1.grid(True)

        ax2.plot(x[:len(ipr_values_plot)], ipr_values_plot,
                 marker='o', markersize=4, color=COLORS[0], label="Avg. IPR")
        ax2.set_title('Sparsity Level of Frequency', fontsize=16)
        ax2.set_xlabel('Step', fontsize=16)
        ax2.legend(fontsize=18, loc="lower right")
        ax2.grid(True)

        _save_fig(fig, self._out('loss_sparsity.png'))

        # --- JSON for interactive Plotly ---
        payload = {
            'epochs': x.tolist(),
            'ipr': ipr_values_plot,
        }
        if loss_values is not None:
            payload['train_loss'] = [float(v) for v in loss_values]
        if curves and 'test_losses' in curves:
            payload['test_loss'] = curves['test_losses']
        with open(self._out('loss_sparsity.json'), 'w') as f:
            json.dump(payload, f)

        print("    Saved loss_sparsity.png, loss_sparsity.json")

    # ------------------------------------------------------------------
    # Tab 2: Fourier Weights (heatmap + lineplots)
    # ------------------------------------------------------------------

    def generate_tab2(self):
        """Generate full_training_para_origin.png, lineplot_in.png, lineplot_out.png."""
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
        W_in_data = model_load['mlp.W_in'][sorted_indices, :]
        W_out_T_data = model_load['mlp.W_out'].T[sorted_indices, :]

        # Sort within selected set by frequency
        sort_order = np.argsort(freq_ls)
        ranked_W_in = W_in_data[sort_order, :]
        ranked_W_out_T = W_out_T_data[sort_order, :]

        # ---- Heatmap plot ----
        fig, axes = plt.subplots(
            2, 1, figsize=(5, 5), constrained_layout=True,
            gridspec_kw={"hspace": 0.05}
        )

        # W_in
        ax_in = axes[0]
        abs_max_in = np.abs(ranked_W_in.detach().cpu().numpy()).max()
        im_in = ax_in.imshow(
            ranked_W_in.detach().cpu().numpy(),
            cmap=CMAP_DIVERGING, vmin=-abs_max_in, vmax=abs_max_in,
            aspect='auto'
        )
        ax_in.set_title(r'First-Layer Parameters $\theta_m$', fontsize=18)
        fig.colorbar(im_in, ax=ax_in)
        y_locs = np.arange(num_neurons)
        ax_in.set_yticks(y_locs)
        ax_in.set_yticklabels(y_locs, fontsize=11)
        ax_in.set_ylabel('Neuron #', fontsize=16)
        x_locs = np.arange(ranked_W_in.shape[1])
        ax_in.set_xticks(x_locs)
        ax_in.set_xticklabels(x_locs, rotation=90, fontsize=11)

        # W_out.T
        ax_out = axes[1]
        abs_max_out = np.abs(ranked_W_out_T.detach().cpu().numpy()).max()
        im_out = ax_out.imshow(
            ranked_W_out_T.detach().cpu().numpy(),
            cmap=CMAP_DIVERGING, vmin=-abs_max_out, vmax=abs_max_out,
            aspect='auto'
        )
        ax_out.set_title(r'Second-Layer Parameters $\xi_m$', fontsize=18)
        fig.colorbar(im_out, ax=ax_out)
        ax_out.set_yticks(y_locs)
        ax_out.set_yticklabels(y_locs, fontsize=11)
        ax_out.set_ylabel('Neuron #', fontsize=16)
        ax_out.set_xticks(x_locs)
        ax_out.set_xticklabels(x_locs, rotation=90, fontsize=11)
        ax_out.set_xlabel('Input / Output Dimension', fontsize=16)

        _save_fig(fig, self._out('full_training_para_origin.png'))

        # ---- Line plots ----
        lineplot_idx = select_lineplot_neurons(list(range(num_neurons)), n=3)
        fb = self.fourier_basis
        positions = np.arange(ranked_W_in.shape[1])

        for tag, weight_data, title_tex in [
            ('lineplot_in', ranked_W_in, r'First-Layer Parameters $\theta_m$'),
            ('lineplot_out', ranked_W_out_T, r'Second-Layer Parameters $\xi_m$'),
        ]:
            if hasattr(weight_data, 'detach'):
                weight_np = weight_data.detach().cpu()
            else:
                weight_np = weight_data

            top3 = weight_np[lineplot_idx]

            fig, axes_lp = plt.subplots(
                nrows=3, ncols=1, figsize=(5, 5),
                constrained_layout=True,
                gridspec_kw={'hspace': 0.02}
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
                ax.set_ylabel(f'Neuron #{i+1}', fontsize=16)
                ax.set_xticks(positions)
                ax.grid(True, which='major', axis='both',
                        linestyle='--', linewidth=0.5, alpha=0.6)
                if i < len(axes_lp) - 1:
                    ax.set_xticklabels([])

            axes_lp[-1].set_xlabel('Input Dimension', fontsize=16)
            axes_lp[-1].set_xticks(positions)
            axes_lp[-1].set_xticklabels(
                np.arange(ranked_W_in.shape[1]), rotation=90, fontsize=11
            )
            axes_lp[0].set_title(title_tex, fontsize=18)
            axes_lp[0].legend(fontsize=14, loc="upper right")

            _save_fig(fig, self._out(f'{tag}.png'))

        print("    Saved full_training_para_origin.png, lineplot_in.png, lineplot_out.png")

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

        # Filter out edge cases near +/-pi boundary
        cond = (((coeff_2phi_arr / (coeff_psi_arr + 1e-12) < 0)
                 & (coeff_2phi_arr < -2.8))
                | ((coeff_2phi_arr / (coeff_psi_arr + 1e-12) < 0)
                   & (coeff_2phi_arr > 2.8)))

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(
            coeff_2phi_arr[~cond], coeff_psi_arr[~cond],
            marker='.', color=COLORS[0], s=20
        )
        x_min, x_max = ax.get_xlim()
        x_line = np.linspace(x_min, x_max, 200)
        ax.plot(x_line, x_line, linestyle='--', color='gray', alpha=0.7)
        ax.set_xlabel(r'Normalized $2\phi_m$', fontsize=19)
        ax.set_ylabel(r'$\psi_m$', fontsize=19)
        ax.set_ylim(-np.pi * 1.1, np.pi * 1.1)
        ax.grid(True)

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
        # Reconstruct EmbedMLP with act_type matching standard run (ReLU)
        # Use Quad activation because the notebook uses Quad for the output logits
        # visualization (as the ReLU model achieves 100% with |x| and x^2).
        # Actually, the notebook cell-22 uses act_type="Quad" to show the logit structure.
        act_type = TRAINING_RUNS['standard']['act_type']
        model = EmbedMLP(
            d_vocab=self.d_vocab,
            d_model=self.d_model,
            d_mlp=self.d_mlp,
            act_type="Quad",  # Matches notebook cell-22 which uses Quad for visualization
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
        if os.path.exists(train_data_path):
            train_data = torch.load(train_data_path, weights_only=True,
                                    map_location=self.device)
        if os.path.exists(test_data_path):
            test_data = torch.load(test_data_path, weights_only=True,
                                   map_location=self.device)

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

        # ---- Loss JSON ----
        if train_losses:
            loss_data = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'stage1_end': stage1_end,
                'stage2_end': stage2_end,
            }
            with open(self._out('grokk_loss.json'), 'w') as f:
                json.dump(loss_data, f)

        # ---- Accuracy: compute from checkpoints if not in curves ----
        train_accs = []
        test_accs = []
        if train_data is not None and test_data is not None:
            train_labels = torch.tensor(
                [(i.item() + j.item()) % p for i, j in train_data],
                dtype=torch.long
            )
            test_labels = torch.tensor(
                [(i.item() + j.item()) % p for i, j in test_data],
                dtype=torch.long
            )
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

        # ---- Phase difference |sin(D*)| ----
        abs_phase_diff = []
        sparse_level = []

        for ep in epochs:
            model_sd = checkpoints[ep]
            W_in_d, W_out_d, mfl = decode_weights(model_sd, self.fourier_basis)

            sparse_level.append(
                ((W_in_d.norm(p=4, dim=1) ** 4 / W_in_d.norm(p=2, dim=1) ** 4).mean() / 2
                 + (W_out_d.norm(p=4, dim=1) ** 4 / W_out_d.norm(p=2, dim=1) ** 4).mean() / 2
                 ).item()
            )

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
        ax1.set_ylim([0, 0.65])

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

        print("    Saved grokk_loss.json, grokk_acc.json, grokk_abs_phase_diff.png, "
              "grokk_avg_ipr.png, grokk_memorization_accuracy.png, "
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

            # Select a neuron that shows clear phase alignment
            # Pick neuron with largest final scale
            final_records = [r for r in all_neuron_records if r['epoch'] == epochs[-1]]
            if not final_records:
                continue
            best_neuron = max(final_records, key=lambda r: r['scale_in'])['neuron']

            # Extract trajectory for this neuron
            neuron_records = [r for r in all_neuron_records if r['neuron'] == best_neuron]
            # Remove last few points if noisy (as notebooks do)
            trim = max(0, len(neuron_records) - 4) if prefix == 'relu' else max(0, len(neuron_records) - 14)
            neuron_records = neuron_records[:trim] if trim > 0 else neuron_records

            phi_in_list = [r['phi_in'] for r in neuron_records]
            phi_out_list = [r['phi_out'] for r in neuron_records]
            phi2_in_list = [2 * v for v in phi_in_list]
            scale_in_list = [r['scale_in'] for r in neuron_records]
            scale_out_list = [r['scale_out'] for r in neuron_records]

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
            if prefix == 'quad':
                keys = [0]
                mid = min(epochs, key=lambda e: abs(e - 1000))
                end = epochs[-1]
                if mid not in keys:
                    keys.append(mid)
                if end not in keys:
                    keys.append(end)
            else:
                keys = [0, epochs[-1]]

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
    # Generate all
    # ------------------------------------------------------------------

    def generate_all(self):
        """Generate all tab plots with error handling."""
        print(f"\n{'=' * 60}")
        print(f"Generating plots for p={self.p}")
        print(f"  Input:  {self.input_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"{'=' * 60}")

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

        print(f"\nDone generating plots for p={self.p}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate all model-dependent plots for the HF app.'
    )
    parser.add_argument('--all', action='store_true',
                        help='Generate plots for all primes found in input dir')
    parser.add_argument('--prime', type=int,
                        help='Generate plots for a specific prime')
    parser.add_argument('--input', type=str, default='./trained_models',
                        help='Base input directory containing p_PPP subdirs')
    parser.add_argument('--output', type=str,
                        default='./hf_app/precomputed_results',
                        help='Base output directory for precomputed results')
    args = parser.parse_args()

    if not args.all and args.prime is None:
        parser.error("Specify --all or --prime P")

    if args.prime:
        primes = [args.prime]
    else:
        # Discover primes from input directory
        primes = []
        if os.path.isdir(args.input):
            for d in sorted(os.listdir(args.input)):
                if d.startswith('p_'):
                    try:
                        p = int(d.split('_')[1])
                        primes.append(p)
                    except (ValueError, IndexError):
                        pass
        if not primes:
            print(f"No p_PPP directories found in {args.input}")
            sys.exit(1)

    total = len(primes)
    for i, p in enumerate(primes):
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
