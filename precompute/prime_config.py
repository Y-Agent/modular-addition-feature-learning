"""
Configuration for all moduli and training runs.
Defines the d_mlp sizing formula and the 5 training run configurations.
p can be any odd number >= 3 (not restricted to primes).
"""
import math


def get_moduli(low=3, high=199):
    """Return all odd numbers in [low, high]."""
    moduli = []
    for n in range(low, high + 1):
        if n >= 3 and n % 2 == 1:
            moduli.append(n)
    return moduli


# Keep old name as alias for backward compatibility
get_primes = get_moduli


def compute_d_mlp(p: int) -> int:
    """
    Compute d_mlp maintaining the ratio from p=23, d_mlp=512.
    Formula: d_mlp = max(512, ceil(512/529 * p^2))
    Can have more neurons but not less than the ratio dictates.
    """
    ratio = 512 / (23 ** 2)  # 512/529 ≈ 0.9679
    return max(512, math.ceil(ratio * p * p))


# Minimum p overall (p=2 has 0 non-DC frequencies, making Fourier analysis degenerate)
MIN_P = 3

# Minimum p for grokking experiments (need enough test data for meaningful split)
MIN_P_GROKKING = 19

# Backward-compatible aliases
MIN_PRIME = MIN_P
MIN_PRIME_GROKKING = MIN_P_GROKKING

# 5 training run configurations per p
TRAINING_RUNS = {
    "standard": {
        "embed_type": "one_hot",
        "init_type": "random",
        "optimizer": "AdamW",
        "act_type": "ReLU",
        "lr": 5e-5,
        "weight_decay": 0,
        "frac_train": 1.0,
        "num_epochs": 5000,
        "save_every": 200,
        "init_scale": 0.1,
        "save_models": True,
        "batch_style": "full",
        "seed": 42,
    },
    "grokking": {
        "embed_type": "one_hot",
        "init_type": "random",
        "optimizer": "AdamW",
        "act_type": "ReLU",
        "lr": 1e-4,
        "weight_decay": 2.0,
        "frac_train": 0.75,
        "num_epochs": 50000,
        "save_every": 200,
        "init_scale": 0.1,
        "save_models": True,
        "batch_style": "full",
        "seed": 42,
    },
    "quad_random": {
        "embed_type": "one_hot",
        "init_type": "random",
        "optimizer": "AdamW",
        "act_type": "Quad",
        "lr": 5e-5,
        "weight_decay": 0,
        "frac_train": 1.0,
        "num_epochs": 5000,
        "save_every": 200,
        "init_scale": 0.1,
        "save_models": True,
        "batch_style": "full",
        "seed": 42,
    },
    "quad_single_freq": {
        "embed_type": "one_hot",
        "init_type": "single-freq",
        "optimizer": "SGD",
        "act_type": "Quad",
        "lr": 0.1,
        "weight_decay": 0,
        "frac_train": 1.0,
        "num_epochs": 10000,
        "save_every": 200,
        "init_scale": 0.02,
        "save_models": True,
        "batch_style": "full",
        "seed": 42,
    },
    "relu_single_freq": {
        "embed_type": "one_hot",
        "init_type": "single-freq",
        "optimizer": "SGD",
        "act_type": "ReLU",
        "lr": 0.01,
        "weight_decay": 0,
        "frac_train": 1.0,
        "num_epochs": 10000,
        "save_every": 200,
        "init_scale": 0.002,
        "save_models": True,
        "batch_style": "full",
        "seed": 42,
    },
}

# Analytical computation configs (no training needed)
ANALYTICAL_CONFIGS = {
    "decouple_dynamics": {
        "init_k": 2,
        "num_steps_case1": 1400,
        "learning_rate_case1": 1,
        "init_phi_case1": 1.5,
        "init_psi_case1": 0.18,
        "num_steps_case2": 700,
        "learning_rate_case2": 1,
        "init_phi_case2": -0.72,
        "init_psi_case2": -2.91,
        "amplitude": 0.02,
    },
}
