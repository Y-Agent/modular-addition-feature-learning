#!/usr/bin/env python3
"""
Generate "Decoupled Simulation" plots -- analytical gradient flow simulations
that don't require trained models.

Produces 2 plots per p, saved to {output_dir}/p_{p:03d}/:
  1. p{p:03d}_phase_align_approx1.png  -- case 1: longer simulation with annotations
  2. p{p:03d}_phase_align_approx2.png  -- case 2: shorter simulation

Usage:
    python generate_analytical.py --all
    python generate_analytical.py --p 23
    python generate_analytical.py --p 23 --output ./my_output
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mechanism_base import get_fourier_basis, normalize_to_pi
from prime_config import get_moduli, ANALYTICAL_CONFIGS

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
COLORS = ['#0D2758', '#60656F', '#DEA54B', '#A32015', '#347186']
DPI = 150


# ===========================================================================
# Decouple dynamics simulation
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

    # Phase wrapping fix: normalize 2*phi to [-pi,pi], adjust psi to
    # stay within pi of 2*phi, then unwrap the time series so there
    # are no discontinuous jumps at +-pi boundaries.
    def _fix_phase_pair(two_phi_raw, psi_raw):
        two_phi = normalize_to_pi(two_phi_raw)
        psi_fixed = normalize_to_pi(psi_raw).copy()
        diff = psi_fixed - two_phi
        psi_fixed[diff > np.pi] -= 2 * np.pi
        psi_fixed[diff < -np.pi] += 2 * np.pi
        return np.unwrap(two_phi), np.unwrap(psi_fixed)

    phis2_plot, psis_plot = _fix_phase_pair(2 * phis, psis)

    x = np.arange(num_steps + 1) * lr
    vline_kwargs = dict(color='gray', linestyle='--', linewidth=1.5)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    fig.suptitle(f'Decoupled Gradient Flow (p={p})', fontsize=20, y=1.01)

    # --- Top: phase alignment ---
    for k in range(1, (p - 1) // 2 + 1):
        if k != init_k:
            bg_2phi, bg_psi = _fix_phase_pair(
                2 * np.array(results['phis'][k]),
                np.array(results['psis'][k]),
            )
            ax1.plot(x, bg_psi, lw=1.5, alpha=0.4, color='gray')
            ax1.plot(x, bg_2phi, lw=1.5, alpha=0.4, color='gray',
                     linestyle='--')
    ax1.plot(x, psis_plot, color=COLORS[3], linewidth=2.5,
             label=r"$\psi_{k^\star}$")
    ax1.plot(x, phis2_plot, linewidth=2.5, color=COLORS[0],
             label=r"$2\phi_{k^\star}$")
    if show_vline:
        ax1.axvline(x=vline_x, **vline_kwargs)
    ax1.set_title('Dynamics of Phase Alignment', fontsize=18)
    ax1.set_ylabel('Phase (radians)', fontsize=14)
    ax1.legend(fontsize=18)
    ax1.grid(True)

    # --- Middle: phase difference ---
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
    max_freq = (p - 1) // 2
    if max_freq < 1:
        print(f"  SKIP: p={p} has no non-DC frequencies for analytical simulation")
        return

    cfg = ANALYTICAL_CONFIGS["decouple_dynamics"]
    device = torch.device("cpu")
    init_k = min(cfg["init_k"], max_freq)
    amplitude = cfg["amplitude"]

    # Case 1: longer simulation with vline annotations
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
        save_path=os.path.join(output_dir, f"p{p:03d}_phase_align_approx1.png"),
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
        save_path=os.path.join(output_dir, f"p{p:03d}_phase_align_approx2.png"),
        show_vline=False,
    )


# ===========================================================================
# Entry point
# ===========================================================================

def generate_all_for_prime(p, output_base):
    """Generate the 2 decoupled simulation plots for a single prime."""
    output_dir = os.path.join(output_base, f"p_{p:03d}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating decoupled simulation plots for p={p}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Use float64 globally for numerical precision in simulations
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    try:
        generate_decouple_dynamics(p, output_dir)
    finally:
        torch.set_default_dtype(prev_dtype)

    print(f"[DONE] p={p}: 2 plots written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate decoupled simulation plots (analytical, no model needed)'
    )
    parser.add_argument('--all', action='store_true',
                        help='Generate plots for all odd p in [3, 199]')
    parser.add_argument('--p', type=int,
                        help='Generate plots for a specific p')
    parser.add_argument('--output', type=str, default='./precomputed_results',
                        help='Base output directory (default: ./precomputed_results)')
    args = parser.parse_args()

    if not args.all and args.p is None:
        parser.error("Specify --all or --p P")

    moduli = [args.p] if args.p else get_moduli()

    for p in moduli:
        generate_all_for_prime(p, args.output)

    print(f"\nAll done. Processed {len(moduli)} value(s) of p.")


if __name__ == "__main__":
    main()
