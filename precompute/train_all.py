#!/usr/bin/env python3
"""
Batch training script for all primes 3-199.

Usage:
    # Train all runs for all primes
    python train_all.py --all

    # Train specific prime
    python train_all.py --prime 23

    # Train specific run type for a prime
    python train_all.py --prime 23 --run standard

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
from prime_config import get_primes, compute_d_mlp, TRAINING_RUNS
from utils import Config
from nnTrainer import Trainer


def build_config_dict(p, run_params):
    """Build a nested config dict compatible with the Config class."""
    d_mlp = compute_d_mlp(p)
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


def run_training(p, run_name, output_base):
    """Train a single run for a single prime."""
    run_params = TRAINING_RUNS[run_name]
    config_dict = build_config_dict(p, run_params)
    d_mlp = compute_d_mlp(p)

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

    trainer.initial_save_if_appropriate()

    for epoch in range(config.num_epochs):
        train_loss, test_loss = trainer.do_a_training_step(epoch)

        if test_loss.item() < config.stopping_thresh:
            print(f"  Early stopping at epoch {epoch}: "
                  f"test loss {test_loss.item():.6f}")
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
                        help='Train all primes and all runs')
    parser.add_argument('--prime', type=int,
                        help='Train a specific prime')
    parser.add_argument('--run', type=str, choices=list(TRAINING_RUNS.keys()),
                        help='Train a specific run type')
    parser.add_argument('--output', type=str, default='./trained_models',
                        help='Output directory for trained models')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed runs (checks DONE marker)')
    args = parser.parse_args()

    if not args.all and args.prime is None:
        parser.error("Specify --all or --prime P")

    primes = [args.prime] if args.prime else get_primes()
    runs = [args.run] if args.run else list(TRAINING_RUNS.keys())

    total = len(primes) * len(runs)
    completed = 0

    for p in primes:
        for run_name in runs:
            completed += 1
            print(f"\n{'='*60}")
            print(f"[{completed}/{total}] p={p}, run={run_name}")
            print(f"{'='*60}")
            try:
                run_training(p, run_name, args.output)
            except Exception as e:
                print(f"[FAIL] p={p}, run={run_name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nAll done. {completed} runs processed.")


if __name__ == "__main__":
    main()
