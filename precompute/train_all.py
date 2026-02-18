#!/usr/bin/env python3
"""
Batch training script for all odd moduli p in [3, 199].

Usage:
    # Train all runs for all odd p
    python train_all.py --all

    # Train specific p
    python train_all.py --p 23

    # Train specific run type for a p
    python train_all.py --p 23 --run standard

    # Resume (skips completed runs)
    python train_all.py --all --resume

    # Custom output directory
    python train_all.py --all --output ./my_models
"""
import argparse
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from prime_config import get_moduli, compute_d_mlp, TRAINING_RUNS, MIN_P, MIN_P_GROKKING
from utils import Config
from nnTrainer import Trainer


def build_config_dict(p, run_params, d_mlp_override=None):
    """Build a nested config dict compatible with the Config class."""
    d_mlp = d_mlp_override if d_mlp_override is not None else compute_d_mlp(p)
    return {
        'data': {
            'p': p,
            'd_vocab': None,
            'fn_name': 'add',
            'frac_train': run_params['frac_train'],
            'batch_style': run_params['batch_style'],
        },
        'model': {
            'd_model': None,
            'd_mlp': d_mlp,
            'act_type': run_params['act_type'],
            'embed_type': run_params['embed_type'],
            'init_type': run_params['init_type'],
            'init_scale': run_params['init_scale'],
        },
        'training': {
            'num_epochs': run_params['num_epochs'],
            'lr': run_params['lr'],
            'weight_decay': run_params['weight_decay'],
            'optimizer': run_params['optimizer'],
            'stopping_thresh': -1,
            'save_models': run_params['save_models'],
            'save_every': run_params['save_every'],
            'seed': run_params['seed'],
        },
    }


def _save_training_log(output_dir, p, run_name, run_params, d_mlp, curves):
    """Save a human-readable training_log.txt summarizing the run."""
    log_path = os.path.join(output_dir, "training_log.txt")
    n_epochs = len(curves.get('train_losses', []))
    with open(log_path, 'w') as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"Training Log: p={p}, run={run_name}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  prime (p)       = {p}\n")
        f.write(f"  d_mlp           = {d_mlp}\n")
        f.write(f"  activation      = {run_params['act_type']}\n")
        f.write(f"  init_type       = {run_params['init_type']}\n")
        f.write(f"  init_scale      = {run_params['init_scale']}\n")
        f.write(f"  optimizer       = {run_params['optimizer']}\n")
        f.write(f"  learning_rate   = {run_params['lr']}\n")
        f.write(f"  weight_decay    = {run_params['weight_decay']}\n")
        f.write(f"  frac_train      = {run_params['frac_train']}\n")
        f.write(f"  num_epochs      = {run_params['num_epochs']}\n")
        f.write(f"  batch_style     = {run_params['batch_style']}\n")
        f.write(f"  seed            = {run_params['seed']}\n")
        f.write(f"\n{'─' * 70}\n")
        f.write(f"{'Epoch':>8s}  {'Train Loss':>12s}  {'Test Loss':>12s}  "
                f"{'Train Acc':>10s}  {'Test Acc':>10s}  "
                f"{'Grad Norm':>10s}  {'Param Norm':>11s}\n")
        f.write(f"{'─' * 70}\n")

        # Print every 100 epochs + the last epoch
        train_losses = curves.get('train_losses', [])
        test_losses = curves.get('test_losses', [])
        train_accs = curves.get('train_accs', [])
        test_accs = curves.get('test_accs', [])
        grad_norms = curves.get('grad_norms', [])
        param_norms = curves.get('param_norms', [])

        step = max(1, n_epochs // 100)  # ~100 lines
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
            f.write(f"{i:>8d}  {tl:>12s}  {tel:>12s}  "
                    f"{ta:>10s}  {tea:>10s}  "
                    f"{gn:>10s}  {pn:>11s}\n")

        f.write(f"{'─' * 70}\n\n")
        f.write(f"Final Results:\n")
        if train_losses:
            f.write(f"  Train Loss  = {train_losses[-1]:.6f}\n")
        if test_losses:
            f.write(f"  Test Loss   = {test_losses[-1]:.6f}\n")
        if train_accs:
            f.write(f"  Train Acc   = {train_accs[-1]:.4f}\n")
        if test_accs:
            f.write(f"  Test Acc    = {test_accs[-1]:.4f}\n")
        if param_norms:
            f.write(f"  Param Norm  = {param_norms[-1]:.4f}\n")
        f.write(f"\nTotal epochs trained: {n_epochs}\n")


def run_training(p, run_name, output_base, d_mlp_override=None):
    """Train a single run for a single prime."""
    if p < MIN_P:
        print(f"[SKIP] p={p}, run={run_name}: p < {MIN_P} (too few Fourier frequencies)")
        return

    # Single-freq init needs at least 1 non-DC frequency: (p-1)//2 >= 1 → p >= 3
    if run_name in ('quad_single_freq', 'relu_single_freq') and (p - 1) // 2 < 1:
        print(f"[SKIP] p={p}, run={run_name}: no non-DC frequencies for single-freq init")
        return

    if run_name == 'grokking' and p < MIN_P_GROKKING:
        print(f"[SKIP] p={p}, run={run_name}: p < {MIN_P_GROKKING} (too few test points)")
        return

    run_params = TRAINING_RUNS[run_name]
    config_dict = build_config_dict(p, run_params, d_mlp_override)
    d_mlp = d_mlp_override if d_mlp_override is not None else compute_d_mlp(p)

    output_dir = os.path.join(output_base, f"p_{p:03d}", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Check if already completed
    marker = os.path.join(output_dir, "DONE")
    if os.path.exists(marker):
        print(f"[SKIP] p={p}, run={run_name} already completed")
        return

    print(f"[TRAIN] p={p}, d_mlp={d_mlp}, run={run_name}, "
          f"epochs={run_params['num_epochs']}")

    config = Config(config_dict)
    trainer = Trainer(config=config, use_wandb=False)

    # Override save directory so checkpoints go into our output structure
    trainer.save_dir = output_dir
    run_subdir = os.path.join(output_dir, trainer.run_name)
    os.makedirs(run_subdir, exist_ok=True)

    # Re-save train/test data to the overridden location so generate_plots.py
    # can find them (Trainer.__init__ saves to the original save_dir)
    torch.save(trainer.train, os.path.join(run_subdir, 'train_data.pth'))
    torch.save(trainer.test, os.path.join(run_subdir, 'test_data.pth'))

    trainer.initial_save_if_appropriate()

    # Plateau early-stopping for grokking: after 10K epochs, if curves
    # haven't changed in the last 1000 epochs, stop training.
    plateau_check = (run_name == 'grokking')
    plateau_min_epoch = 10000
    plateau_window = 1000
    plateau_loss_tol = 1e-3   # absolute change in loss
    plateau_acc_tol = 0.005   # absolute change in accuracy

    for epoch in range(config.num_epochs):
        train_loss, test_loss = trainer.do_a_training_step(epoch)

        if test_loss.item() < config.stopping_thresh:
            print(f"  Early stopping at epoch {epoch}: "
                  f"test loss {test_loss.item():.6f}")
            break

        # Plateau detection for grokking
        if (plateau_check and epoch >= plateau_min_epoch
                and epoch % plateau_window == 0):
            tl = trainer.train_losses
            tel = trainer.test_losses
            ta = trainer.train_accs
            tea = trainer.test_accs
            w = plateau_window
            if len(tl) >= w and len(tel) >= w:
                tl_flat = (max(tl[-w:]) - min(tl[-w:])) < plateau_loss_tol
                tel_flat = (max(tel[-w:]) - min(tel[-w:])) < plateau_loss_tol
                ta_flat = (not ta) or (max(ta[-w:]) - min(ta[-w:])) < plateau_acc_tol
                tea_flat = (not tea) or (max(tea[-w:]) - min(tea[-w:])) < plateau_acc_tol
                if tl_flat and tel_flat and ta_flat and tea_flat:
                    print(f"  Plateau early stopping at epoch {epoch}: "
                          f"no change in last {w} epochs")
                    break

        if config.is_it_time_to_save(epoch=epoch):
            trainer.save_epoch(epoch=epoch, save_to_wandb=False, local_save=True)

    trainer.post_training_save(
        save_optimizer_and_scheduler=False, log_to_wandb=False
    )

    # Save training curves as JSON for plot generation
    curves = {
        'train_losses': trainer.train_losses,
        'test_losses': trainer.test_losses,
        'train_accs': trainer.train_accs,
        'test_accs': trainer.test_accs,
        'grad_norms': trainer.grad_norms,
        'param_norms': trainer.param_norms,
    }
    curves_path = os.path.join(output_dir, "training_curves.json")
    with open(curves_path, 'w') as f:
        json.dump(curves, f)

    # Save a human-readable training log
    _save_training_log(output_dir, p, run_name, run_params, d_mlp, curves)

    # Write completion marker
    with open(marker, 'w') as f:
        f.write(f"p={p} run={run_name} completed\n")

    print(f"[DONE] p={p}, run={run_name}, "
          f"train_acc={trainer.train_accs[-1]:.4f}, "
          f"test_acc={trainer.test_accs[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch training for modular addition experiments'
    )
    parser.add_argument('--all', action='store_true',
                        help='Train all odd p in [3, 199]')
    parser.add_argument('--p', type=int,
                        help='Train a specific odd modulus p')
    parser.add_argument('--run', type=str, choices=list(TRAINING_RUNS.keys()),
                        help='Train a specific run type')
    parser.add_argument('--output', type=str, default='./trained_models',
                        help='Output directory for trained models')
    parser.add_argument('--d_mlp', type=int, default=None,
                        help='Override d_mlp (number of hidden neurons). '
                             'Default: auto-computed from p.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed runs (checks DONE marker)')
    args = parser.parse_args()

    if not args.all and args.p is None:
        parser.error("Specify --all or --p P")

    moduli = [args.p] if args.p else get_moduli()
    runs = [args.run] if args.run else list(TRAINING_RUNS.keys())

    total = len(moduli) * len(runs)
    completed = 0

    for p in moduli:
        for run_name in runs:
            completed += 1
            print(f"\n{'='*60}")
            print(f"[{completed}/{total}] p={p}, run={run_name}")
            print(f"{'='*60}")
            try:
                run_training(p, run_name, args.output, d_mlp_override=args.d_mlp)
            except Exception as e:
                print(f"[FAIL] p={p}, run={run_name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nAll done. {completed} runs processed.")


if __name__ == "__main__":
    main()
