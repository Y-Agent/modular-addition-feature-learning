#!/usr/bin/env python3
"""
Generate Tab 8 "Theory" plots -- analytical/simulation plots that don't require
trained models.

Produces 4 plots per prime, saved to {output_dir}/p_{p:03d}/tab8_theory/:
  1. phase_align_approx1.png  -- decouple dynamics simulation (case 1)
  2. phase_align_approx2.png  -- decouple dynamics simulation (case 2)
  3. frequency_diversity_output_logits.png -- frequency diversity analysis
  4. phase_diversity_output_logits.png     -- phase diversity analysis

Usage:
    python generate_analytical.py --all
    python generate_analytical.py --prime 23
    python generate_analytical.py --prime 23 --output ./my_output
"""
import argparse
import os
import sys
import random
from itertools import combinations

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mechanism_base import get_fourier_basis, normalize_to_pi
from prime_config import get_primes, ANALYTICAL_CONFIGS

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
COLORS = ['#0D2758', '#60656F', '#DEA54B', '#A32015', '#347186']
DPI = 150


# ===========================================================================
# 1.  Decouple dynamics simulation
# ===========================================================================

def gradient_update(theta, xi, p, device):
    """
    Compute the sum of gradients over all frequency modes k.

    For each frequency k from 1 to (p-1)//2, project theta and xi onto the
    Fourier basis to obtain 2-coefficient vectors, then compute alpha, phi,
    beta, psi and the corresponding gradient contributions.
    """
    fourier_basis, _ = get_fourier_basis(p, device)
    fourier_basis = fourier_basis.to(theta.dtype)
    theta_coeff = fourier_basis @ theta
    xi_coeff = fourier_basis @ xi

    total_grad_theta = torch.zeros_like(theta)
    total_grad_xi = torch.zeros_like(xi)

    j_values = torch.arange(p, device=device, dtype=theta.dtype)
    factor = np.sqrt(2.0 / p)

    for k in range(1, p // 2 + 1):
        coeff_indices = [k * 2 - 1, k * 2]
        neuron_coeff_theta = theta_coeff[coeff_indices]
        neuron_coeff_xi = xi_coeff[coeff_indices]

        alpha = factor * torch.norm(neuron_coeff_theta, dim=0)
        phi = torch.arctan2(-neuron_coeff_theta[1], neuron_coeff_theta[0])

        beta = factor * torch.norm(neuron_coeff_xi, dim=0)
        psi = torch.arctan2(-neuron_coeff_xi[1], neuron_coeff_xi[0])

        w_k = 2 * np.pi * k / p
        grad_theta_k = 2 * p * alpha * beta * torch.cos(w_k * j_values + psi - phi)
        grad_xi_k = p * alpha.pow(2) * torch.cos(w_k * j_values + 2 * phi)

        total_grad_theta += grad_theta_k / p ** 2
        total_grad_xi += grad_xi_k / p ** 2

    return total_grad_theta, total_grad_xi


def simulate_gradient_flow(theta, xi, p, num_steps, learning_rate, device):
    """Euler integration of the coupled gradient-flow ODEs."""
    theta_history = [theta.clone()]
    xi_history = [xi.clone()]

    for _ in range(num_steps):
        grad_theta, grad_xi = gradient_update(theta, xi, p, device)
        theta = theta + learning_rate * grad_theta
        xi = xi + learning_rate * grad_xi
        theta_history.append(theta.clone())
        xi_history.append(xi.clone())

    return theta_history, xi_history


def analyze_history(theta_history, xi_history, p, fourier_basis):
    """
    Extract time series of alpha, phi, beta, psi, delta for every frequency k.
    """
    theta_hist_tensor = torch.stack(theta_history)
    xi_hist_tensor = torch.stack(xi_history)

    theta_coeffs_hist = fourier_basis @ theta_hist_tensor.T
    xi_coeffs_hist = fourier_basis @ xi_hist_tensor.T

    results = {
        'alphas': {}, 'phis': {}, 'betas': {}, 'psis': {}, 'deltas': {}
    }
    factor = np.sqrt(2.0 / p)

    for k in range(1, p // 2 + 1):
        idx = [k * 2 - 1, k * 2]
        neuron_theta_hist = theta_coeffs_hist[idx, :]
        neuron_xi_hist = xi_coeffs_hist[idx, :]

        alphas_k = factor * torch.norm(neuron_theta_hist, dim=0)
        phis_k = torch.atan2(-neuron_theta_hist[1, :], neuron_theta_hist[0, :])

        betas_k = factor * torch.norm(neuron_xi_hist, dim=0)
        psis_k = torch.atan2(-neuron_xi_hist[1, :], neuron_xi_hist[0, :])

        deltas_k = normalize_to_pi(2 * phis_k - psis_k)

        results['alphas'][k] = alphas_k.numpy()
        results['phis'][k] = phis_k.numpy()
        results['betas'][k] = betas_k.numpy()
        results['psis'][k] = psis_k.numpy()
        results['deltas'][k] = deltas_k.numpy()

    return results


def _run_decouple_simulation(p, init_k, num_steps, lr, init_phi, init_psi,
                             amplitude, device):
    """Initialize and run a single decouple-dynamics simulation."""
    fourier_basis, _ = get_fourier_basis(p, device)
    fourier_basis = fourier_basis.to(torch.float64)
    w_k = 2 * np.pi * init_k / p

    theta_init = amplitude * torch.tensor(
        [np.cos(w_k * j + init_phi) for j in range(p)],
        device=device, dtype=torch.float64,
    )
    xi_init = amplitude * torch.tensor(
        [np.cos(w_k * j + init_psi) for j in range(p)],
        device=device, dtype=torch.float64,
    )

    theta_history, xi_history = simulate_gradient_flow(
        theta_init, xi_init, p, num_steps, lr, device,
    )
    results = analyze_history(theta_history, xi_history, p, fourier_basis)
    return results


def _plot_decouple(results, p, num_steps, lr, init_k, save_path,
                   show_vline=True, vline_x=500):
    """
    Publication-quality 3-panel figure:
      Top:    psi_k* and 2*phi_k* vs time
      Middle: D_k* (phase difference) vs time, horizontal line at pi/2
      Bottom: alpha_k* and beta_k* vs time
    """
    plt.rcParams['mathtext.fontset'] = 'cm'

    alphas = np.array(results['alphas'][init_k])
    betas = np.array(results['betas'][init_k])
    deltas = np.array(results['deltas'][init_k])
    phis = np.array(results['phis'][init_k])
    psis = np.array(results['psis'][init_k])

    x = np.arange(num_steps + 1) * lr
    vline_kwargs = dict(color='gray', linestyle='--', linewidth=1.5)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    # --- Top: phase alignment ---
    # Gray background lines for other frequencies
    for k in range(1, (p - 1) // 2 + 1):
        if k != init_k:
            ax1.plot(x, np.array(results['psis'][k]),
                     lw=1.5, alpha=0.4, color='gray')
            ax1.plot(x, 2 * np.array(results['phis'][k]),
                     lw=1.5, alpha=0.4, color='gray', linestyle='--')
    ax1.plot(x, psis, color=COLORS[3], linewidth=2.5,
             label=r"$\psi_{k^\star}$")
    ax1.plot(x, phis * 2, linewidth=2.5, color=COLORS[0],
             label=r"$2\phi_{k^\star}$")
    if show_vline:
        ax1.axvline(x=vline_x, **vline_kwargs)
    ax1.set_title('Dynamics of Phase Alignment', fontsize=18)
    ax1.set_ylabel('Phase (radians)', fontsize=14)
    ax1.legend(fontsize=18)
    ax1.grid(True)

    # --- Middle: phase difference ---
    # Gray background lines for other frequencies
    for k in range(1, (p - 1) // 2 + 1):
        if k != init_k:
            ax2.plot(x, np.array(results['deltas'][k]),
                     lw=1.5, alpha=0.4, color='gray')
    ax2.plot(x, deltas, color=COLORS[0], linewidth=2.5,
             label=r"$D_{k^\star}$")
    if show_vline:
        ax2.axvline(x=vline_x, **vline_kwargs)
        ax2.axhline(y=np.pi / 2, **vline_kwargs)
        ax2.text(x=max(x) * 0.05, y=np.pi / 2 - 0.45,
                 s=r"$D^\star_{k^\star}=\pi/2$", fontsize=16, color='black')
    ax2.set_title('Dynamics of Phase Difference', fontsize=18)
    ax2.set_ylabel('Phase (radians)', fontsize=14)
    ax2.legend(fontsize=18)
    ax2.grid(True)

    # --- Bottom: magnitude evolution ---
    # Gray background lines for other frequencies
    for k in range(1, (p - 1) // 2 + 1):
        if k != init_k:
            ax3.plot(x, np.array(results['alphas'][k]),
                     lw=1.5, alpha=0.4, color='gray')
            ax3.plot(x, np.array(results['betas'][k]),
                     lw=1.5, alpha=0.4, color='gray', linestyle='--')
    ax3.plot(x, alphas, linewidth=2.5, color=COLORS[0],
             label=r"$\alpha_{k^\star}$")
    ax3.plot(x, betas, linewidth=2.5, color=COLORS[3],
             label=r"$\beta_{k^\star}$")
    if show_vline:
        ax3.axvline(x=vline_x, **vline_kwargs)
    ax3.set_title('Magnitude Evolution', fontsize=18)
    ax3.set_xlabel('Time', fontsize=18)
    ax3.set_ylabel('Magnitude', fontsize=14)
    ax3.legend(fontsize=18)
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def generate_decouple_dynamics(p, output_dir):
    """Generate the two decouple-dynamics phase-alignment plots."""
    cfg = ANALYTICAL_CONFIGS["decouple_dynamics"]
    device = torch.device("cpu")
    init_k = min(cfg["init_k"], (p - 1) // 2)
    amplitude = cfg["amplitude"]

    # Case 1: longer simulation with vline
    print(f"  Running decouple dynamics case 1 (p={p}) ...")
    results1 = _run_decouple_simulation(
        p, init_k,
        num_steps=cfg["num_steps_case1"],
        lr=cfg["learning_rate_case1"],
        init_phi=cfg["init_phi_case1"],
        init_psi=cfg["init_psi_case1"],
        amplitude=amplitude,
        device=device,
    )
    _plot_decouple(
        results1, p,
        num_steps=cfg["num_steps_case1"],
        lr=cfg["learning_rate_case1"],
        init_k=init_k,
        save_path=os.path.join(output_dir, "phase_align_approx1.png"),
        show_vline=True,
        vline_x=cfg["num_steps_case1"] * cfg["learning_rate_case1"] * 0.36,
    )

    # Case 2: shorter simulation without vline annotations
    print(f"  Running decouple dynamics case 2 (p={p}) ...")
    results2 = _run_decouple_simulation(
        p, init_k,
        num_steps=cfg["num_steps_case2"],
        lr=cfg["learning_rate_case2"],
        init_phi=cfg["init_phi_case2"],
        init_psi=cfg["init_psi_case2"],
        amplitude=amplitude,
        device=device,
    )
    _plot_decouple(
        results2, p,
        num_steps=cfg["num_steps_case2"],
        lr=cfg["learning_rate_case2"],
        init_k=init_k,
        save_path=os.path.join(output_dir, "phase_align_approx2.png"),
        show_vline=False,
    )


# ===========================================================================
# 2.  Frequency / Phase Diversity Analysis
# ===========================================================================

def _build_neuron_counts(p, neuron_budget):
    """
    Distribute neuron_budget evenly across (p-1)//2 frequencies.
    Returns an array of length num_freqs.
    """
    num_freqs = (p - 1) // 2
    neurons_per_freq = neuron_budget // num_freqs
    remaining = neuron_budget % num_freqs
    neuron_counts = np.array([neurons_per_freq] * num_freqs)
    neuron_counts[:remaining] += 1
    return neuron_counts


def compute_logits_vectorized(x, y, freq_indices, neuron_counts, p, a=1.0):
    """
    Compute analytical output logits for a given (x, y) pair using specified
    frequencies.  Phases are deterministically spaced on [0, 2*pi).
    """
    logits = np.zeros(p)
    z_vals = np.arange(p)

    for k_idx in freq_indices:
        k = k_idx
        omega_k = 2 * np.pi * k / p
        num_neurons = neuron_counts[k_idx - 1]
        phis = np.linspace(0, 2 * np.pi, num_neurons, endpoint=False)

        for phi in phis:
            input_term = np.cos(omega_k * x + phi) + np.cos(omega_k * y + phi)
            output_terms = np.cos(omega_k * z_vals + 2 * phi)
            logits += a * output_terms * (input_term ** 2)

    return logits


def compute_logits_with_phase_range(x, y, freq_indices, neuron_counts, p,
                                    phase_range, a=1.0):
    """
    Same as compute_logits_vectorized but phases are spaced on [0, phase_range).
    """
    logits = np.zeros(p)
    z_vals = np.arange(p)

    for k_idx in freq_indices:
        k = k_idx
        omega_k = 2 * np.pi * k / p
        num_neurons = neuron_counts[k_idx - 1]
        phis = np.linspace(0, phase_range, num_neurons, endpoint=False)

        for phi in phis:
            input_term = np.cos(omega_k * x + phi) + np.cos(omega_k * y + phi)
            output_terms = np.cos(omega_k * z_vals + 2 * phi)
            logits += a * output_terms * (input_term ** 2)

    return logits


def _get_test_pairs(p):
    """
    Return 4 test (x, y) pairs adapted to the prime p.
    """
    return [
        (max(1, p // 4), max(1, p // 4)),
        (max(1, p // 6), max(1, p // 4)),
        (max(1, p // 8), min(p - 1, p // 2)),
        (max(1, p // 5), max(1, p // 4)),
    ]


def _get_freq_combinations(num_freqs_to_select, total_freqs, max_combos=100):
    """Get all combinations of num_freqs_to_select from 1..total_freqs,
    randomly sampled down to max_combos if needed."""
    all_freqs = list(range(1, total_freqs + 1))
    all_combos = list(combinations(all_freqs, num_freqs_to_select))
    if len(all_combos) > max_combos:
        random.seed(42)
        all_combos = random.sample(all_combos, max_combos)
    return all_combos


def generate_frequency_diversity(p, output_dir):
    """
    Generate the frequency diversity analysis plot.

    For each of [1, 2, 4, 8, full] frequency counts, sample random subsets,
    compute analytical logits, and plot mean +/- std for 4 test pairs.
    """
    print(f"  Generating frequency diversity plot (p={p}) ...")
    np.random.seed(42)
    random.seed(42)

    cfg = ANALYTICAL_CONFIGS["frequency_subset"]
    neuron_budget = cfg["neuron_budget"]
    num_freqs = (p - 1) // 2
    neuron_counts = _build_neuron_counts(p, neuron_budget)

    freq_counts = cfg["freq_counts"] + [num_freqs]  # append full
    freq_labels = [rf'${n}$ Freqs' for n in cfg["freq_counts"]]
    freq_labels.append(rf'Full $({num_freqs})$ Freqs')

    # Filter out freq_counts that exceed num_freqs
    valid = [(fc, fl) for fc, fl in zip(freq_counts, freq_labels)
             if fc <= num_freqs]
    freq_counts = [v[0] for v in valid]
    freq_labels = [v[1] for v in valid]

    # Color progression: light blue to dark blue, then red for full
    white_to_blue = LinearSegmentedColormap.from_list(
        'white_blue', ['white', '#0D2758'], N=256)
    n_non_full = len(freq_counts) - 1
    colors_non_full = [white_to_blue(x)
                       for x in np.linspace(0.3, 0.9, max(n_non_full, 1))]
    colors = colors_non_full[:n_non_full] + ['#A32015']

    test_pairs = _get_test_pairs(p)
    z_vals = np.arange(p)

    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    legend_handles = []
    legend_labels = []

    for pair_idx, (x, y) in enumerate(test_pairs):
        ax = axes[pair_idx]
        z_true = (x + y) % p

        for fidx, (n_freq, label) in enumerate(zip(freq_counts, freq_labels)):
            if n_freq == num_freqs:
                freq_combos = [tuple(range(1, num_freqs + 1))]
            else:
                freq_combos = _get_freq_combinations(n_freq, num_freqs,
                                                     max_combos=100)

            all_logits = []
            for combo in freq_combos:
                logits = compute_logits_vectorized(
                    x, y, list(combo), neuron_counts, p)
                all_logits.append(logits)

            all_logits = np.array(all_logits)
            mean_logits = np.mean(all_logits, axis=0)
            std_logits = np.std(all_logits, axis=0)

            line, = ax.plot(z_vals, mean_logits, color=colors[fidx],
                            linewidth=2, label=label, alpha=0.9)
            ax.fill_between(z_vals,
                            mean_logits - std_logits,
                            mean_logits + std_logits,
                            color=colors[fidx], alpha=0.2)

            if pair_idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.axvline(x=z_true, color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Prediction', fontsize=18)
        if pair_idx == 0:
            ax.set_ylabel('Output Logit', fontsize=18)
        ax.set_title(rf'$(x,y)=({x},{y})$', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, p, max(1, p // 6)))
        ax.tick_params(labelsize=11)

    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=5,
               fontsize=18, bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    save_path = os.path.join(output_dir, "frequency_diversity_output_logits.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def generate_phase_diversity(p, output_dir):
    """
    Generate the phase diversity analysis plot.

    Uses all (p-1)//2 frequencies but varies the phase range from
    [0, 0.4*pi] to [0, 2.0*pi].
    """
    print(f"  Generating phase diversity plot (p={p}) ...")
    np.random.seed(42)

    cfg = ANALYTICAL_CONFIGS["frequency_subset"]
    neuron_budget = cfg["neuron_budget"]
    num_freqs = (p - 1) // 2
    neuron_counts = _build_neuron_counts(p, neuron_budget)
    all_freq_indices = list(range(1, num_freqs + 1))

    phase_multipliers = np.array(cfg["phase_multipliers"])
    phase_ranges = phase_multipliers * np.pi
    phase_labels = [rf'$[0, {mult:.2g}\pi]$' for mult in phase_multipliers]

    # Color progression: light blue to dark blue, then red for last (2*pi)
    white_to_blue = LinearSegmentedColormap.from_list(
        'white_blue', ['white', '#0D2758'], N=256)
    n_non_full = len(phase_multipliers) - 1
    colors_non_full = [white_to_blue(x)
                       for x in np.linspace(0.3, 0.9, max(n_non_full, 1))]
    colors = colors_non_full[:n_non_full] + ['#A32015']

    test_pairs = _get_test_pairs(p)
    z_vals = np.arange(p)

    plt.rcParams['mathtext.fontset'] = 'cm'
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    legend_handles = []
    legend_labels = []

    for pair_idx, (x, y) in enumerate(test_pairs):
        ax = axes[pair_idx]
        z_true = (x + y) % p

        for fidx, (phase_range, label) in enumerate(
                zip(phase_ranges, phase_labels)):
            logits = compute_logits_with_phase_range(
                x, y, all_freq_indices, neuron_counts, p, phase_range)

            line, = ax.plot(z_vals, logits, color=colors[fidx],
                            linewidth=2, label=label, alpha=0.9)

            if pair_idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.axvline(x=z_true, color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Prediction', fontsize=18)
        if pair_idx == 0:
            ax.set_ylabel('Output Logit', fontsize=18)
        ax.set_title(rf'$(x,y)=({x},{y})$', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, p, max(1, p // 6)))
        ax.tick_params(labelsize=11)

    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=5,
               fontsize=18, bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    save_path = os.path.join(output_dir, "phase_diversity_output_logits.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ===========================================================================
# 3.  Entry point
# ===========================================================================

def generate_all_for_prime(p, output_base):
    """Generate all 4 Tab-8 theory plots for a single prime."""
    output_dir = os.path.join(output_base, f"p_{p:03d}", "tab8_theory")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating Tab 8 Theory plots for p={p}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Use float64 globally for numerical precision in simulations
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    try:
        generate_decouple_dynamics(p, output_dir)
        generate_frequency_diversity(p, output_dir)
        generate_phase_diversity(p, output_dir)
    finally:
        torch.set_default_dtype(prev_dtype)

    print(f"[DONE] p={p}: 4 plots written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Tab 8 "Theory" analytical/simulation plots'
    )
    parser.add_argument('--all', action='store_true',
                        help='Generate plots for all primes 3-199')
    parser.add_argument('--prime', type=int,
                        help='Generate plots for a specific prime')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Base output directory (default: ./outputs)')
    args = parser.parse_args()

    if not args.all and args.prime is None:
        parser.error("Specify --all or --prime P")

    primes = [args.prime] if args.prime else get_primes()

    for p in primes:
        generate_all_for_prime(p, args.output)

    print(f"\nAll done. Processed {len(primes)} prime(s).")


if __name__ == "__main__":
    main()
