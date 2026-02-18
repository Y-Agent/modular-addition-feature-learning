"""
Automated neuron selection strategies for all primes.
Replaces hard-coded neuron indices from the analysis notebooks.
"""
import torch
import numpy as np
from collections import Counter


def select_top_neurons_by_frequency(max_freq_ls, W_in_decode, n=20):
    """
    Select top N neurons covering all frequencies (round-robin).
    Used for heatmap plots (Tab 2).

    Picks the highest-magnitude neuron from each frequency in turn,
    cycling through frequencies until n neurons are selected. This ensures
    the heatmap shows diversification across all frequencies, matching
    the blog's Figure 2.

    Returns list of neuron indices into the original d_mlp-sized arrays.
    """
    d_mlp = W_in_decode.shape[0]
    magnitudes = W_in_decode.abs().max(dim=1).values

    # Group neurons by their dominant frequency, sorted by magnitude (descending)
    from collections import defaultdict
    freq_groups = defaultdict(list)
    for i in range(d_mlp):
        f = max_freq_ls[i]
        if f > 0:  # skip DC neurons
            freq_groups[f].append((magnitudes[i].item(), i))

    # Sort each group by magnitude descending
    for f in freq_groups:
        freq_groups[f].sort(key=lambda x: -x[0])

    # Round-robin across frequencies (ascending order)
    freqs_sorted = sorted(freq_groups.keys())
    selected = []
    pointers = {f: 0 for f in freqs_sorted}

    while len(selected) < min(n, d_mlp) and freqs_sorted:
        exhausted = []
        for f in freqs_sorted:
            if len(selected) >= n:
                break
            if pointers[f] < len(freq_groups[f]):
                _, idx = freq_groups[f][pointers[f]]
                selected.append(idx)
                pointers[f] += 1
            else:
                exhausted.append(f)
        for f in exhausted:
            freqs_sorted.remove(f)

    return selected


def select_lineplot_neurons(sorted_indices, n=3):
    """
    Select first N neurons from the frequency-sorted set for line plots (Tab 2).
    Picks neurons evenly spaced through the sorted list to show diverse frequencies.
    """
    if len(sorted_indices) <= n:
        return list(range(len(sorted_indices)))
    step = len(sorted_indices) // n
    return [i * step for i in range(n)]


def select_phase_frequency(max_freq_ls, p):
    """
    Choose the frequency for phase distribution analysis (Tab 3).
    Picks the frequency with the most neurons assigned to it (mode),
    excluding frequency 0 (DC component).
    """
    freq_counts = Counter(f for f in max_freq_ls if f > 0)
    if not freq_counts:
        return 1
    return freq_counts.most_common(1)[0][0]


def select_lottery_neuron(model_load, fourier_basis, decode_scales_phis_fn):
    """
    Find the neuron with the clearest frequency specialization (Tab 6).
    Picks the neuron with the highest ratio of dominant frequency scale
    to second-highest frequency scale.
    """
    scales, _, _ = decode_scales_phis_fn(model_load, fourier_basis)
    # scales: [n_neurons, K+1], skip DC at index 0
    scales_no_dc = scales[:, 1:]

    if scales_no_dc.shape[1] < 2:
        return 0

    sorted_scales, _ = torch.sort(scales_no_dc, dim=1, descending=True)
    ratio = sorted_scales[:, 0] / (sorted_scales[:, 1] + 1e-10)

    return ratio.argmax().item()
