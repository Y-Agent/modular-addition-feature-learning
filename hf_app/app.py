#!/usr/bin/env python3
"""
Gradio app for Modular Addition Feature Learning visualization.
Serves pre-computed results for odd moduli p in [3, 199].

All results are pre-computed as PNG images and JSON data files.
No GPU needed at serving time.

Tab structure:
  Core Interpretability:
    1. Training Overview    -- loss + IPR sparsity
    2. Fourier Weights      -- decoded W_in/W_out heatmaps + line plots + neuron inspector
    3. Phase Analysis       -- phase distribution, 2phi vs psi, magnitudes
    4. Output Logits        -- predicted logit heatmap + interactive logit explorer
    5. Lottery Mechanism    -- neuron specialization, magnitude/phase, contour
  Grokking:
    6. Grokking             -- loss/acc, phase alignment, IPR, memorization, epoch slider
  Theory:
    7. Gradient Dynamics    -- phase alignment for Quad & ReLU single-freq init
    8. Decoupled Simulation -- analytical gradient flow (no model needed)
  Diagnostics:
    9. Training Log         -- per-run hyperparameters and epoch-by-epoch metrics
"""
import gradio as gr
import json
import logging
import os
import shutil
import subprocess
import sys

import numpy as np

logger = logging.getLogger(__name__)
# Force pandas to be fully imported before plotly lazily imports it
# (avoids "partially initialized module 'pandas'" in threaded callbacks)
import pandas  # noqa: F401
import plotly.graph_objects as go

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "precomputed_results")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "trained_models")

# Max p for on-demand training (d_mlp grows as O(p^2), memory limit)
MAX_P_ON_DEMAND = 97

COLORS = ['#0D2758', '#60656F', '#DEA54B', '#A32015', '#347186']
STAGE_COLORS = ['rgba(212,175,55,0.15)', 'rgba(139,115,85,0.15)', 'rgba(192,192,192,0.15)']

# KaTeX delimiters for Gradio Markdown
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

# Custom CSS for Palatino font and styling
CUSTOM_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

* {
    font-family: "Palatino Linotype", "Book Antiqua", Palatino, "Libre Baskerville", Georgia, serif !important;
}
code, pre, .code, .monospace {
    font-family: "Menlo", "Consolas", "Monaco", monospace !important;
}
.katex, .katex * {
    font-family: KaTeX_Main, "Times New Roman", serif !important;
}
h1 {
    font-family: "Palatino Linotype", "Book Antiqua", Palatino, "Libre Baskerville", Georgia, serif !important;
    text-align: center !important;
    margin-bottom: 0.1em !important;
}
h3 {
    font-family: "Palatino Linotype", "Book Antiqua", Palatino, "Libre Baskerville", Georgia, serif !important;
    text-align: center !important;
    color: var(--neutral-500) !important;
    font-weight: normal !important;
    margin-top: 0 !important;
}
h2, h4 {
    font-family: "Palatino Linotype", "Book Antiqua", Palatino, "Libre Baskerville", Georgia, serif !important;
}
blockquote {
    border-left: 3px solid var(--color-accent) !important;
    background-color: var(--block-background-fill) !important;
    padding: 0.5em 1em !important;
    margin: 0.5em 0 !important;
}
"""

# ---------------------------------------------------------------------------
# Math explanation text for each tab (following the paper precisely)
# ---------------------------------------------------------------------------

MATH_TAB1 = r"""
### Overview

We study how a two-layer neural network learns to compute modular addition $f(x,y) = (x+y) \bmod p$. The network has $M$ hidden neurons. Each input integer $x$ is represented as a one-hot vector, and the network produces a score for each of the $p$ possible answers. During training, the network learns two weight vectors per neuron: an **input weight** $\theta_m$ and an **output weight** $\xi_m$, both vectors of length $p$.

#### Two Training Setups

1. **Full-data (Tabs 1--5, 7).** Train on all $p^2$ input pairs with no held-out data and no regularization. This produces clean features ideal for studying what the network learns and how.

2. **Grokking (Tab 6).** Train on only 75% of input pairs with weight decay $\lambda = 2.0$ (a penalty that shrinks weights over time). These two ingredients -- incomplete data + weight decay -- cause the network to first memorize, then suddenly generalize, a phenomenon called **grokking**.

#### What the Network Learns

Each neuron's weight vectors turn into **cosine waves** at a single frequency -- the network independently rediscovers the Discrete Fourier Transform. The neurons collectively cover all frequencies with balanced strengths, enabling them to "vote" together and identify the correct answer $(x+y) \bmod p$.

#### How It Learns (Dynamics)

Frequencies **compete** within each neuron during training. The frequency whose input and output phases happen to start best-aligned grows fastest -- a **lottery ticket mechanism** where the random initialization determines the outcome before training begins.

#### Grokking (Three Stages)

When trained on partial data with weight decay: **(I) Memorization** -- the network fits the training data using noisy, multi-frequency features. **(II) Generalization** -- weight decay prunes away the noise, leaving clean single-frequency features; test accuracy jumps. **(III) Cleanup** -- weight decay slowly polishes the solution.

#### Progress Measures on These Plots

- **Loss**: Cross-entropy loss (lower = better predictions). We show both training loss and test loss.

- **IPR (Inverse Participation Ratio)**: Measures how concentrated a neuron's energy is across frequencies. We decompose each neuron's weights into Fourier components, measure the strength $A_k$ at each frequency $k$, and compute:

$$\text{IPR} = \frac{\sum_k A_k^4}{\left(\sum_k A_k^2\right)^2}.$$

When a neuron uses only **one frequency**, IPR $= 1$ (fully specialized). When energy is spread across **many frequencies**, IPR is close to $0$. Watching IPR rise toward 1 during training shows the network specializing.

- **Phase scatter**: Each neuron has an input phase $\phi_m$ and output phase $\psi_m$. The theory predicts the output phase equals twice the input phase ($\psi_m = 2\phi_m$). The scatter plot checks this: all points should fall on the diagonal.
"""

MATH_TAB2 = r"""
### Every Neuron is a Cosine Wave
> **Setup:** ReLU activation, full data, no weight decay.

After training, each neuron's weight vectors become clean **cosine waves** at a single frequency. Concretely, the input weight of neuron $m$ looks like:

$$\underbrace{\theta_m[j]}_{\text{input weight at position } j} = \underbrace{\alpha_m}_{\text{input magnitude}} \cdot \cos\!\left(\underbrace{\frac{2\pi k}{p}}_{\text{frequency}} \cdot j + \underbrace{\phi_m}_{\text{input phase}}\right),$$

and the output weight has the same form with its own magnitude $\beta_m$ (output magnitude) and phase $\psi_m$ (output phase). Each neuron picks **one frequency** $k$ out of the $(p{-}1)/2$ possible frequencies. No one told the network about Fourier analysis -- it rediscovered this representation on its own through training.

**Heatmap**: Each row is a neuron, each column is a Fourier component (cosine and sine at each frequency). If a row has only one bright cell, that neuron is using a single frequency -- and that's exactly what we see.

**Line Plots**: The dots are the actual learned weights; the dashed curves are best-fit cosines. The near-perfect fits confirm each neuron is well-described by a single cosine at a single frequency.

**Neuron Inspector**: Select a neuron from the dropdown to see how its energy is distributed across all frequencies (for both input and output weights).
"""

MATH_TAB3 = r"""
### Phase Alignment and Collective Diversification
> **Setup:** ReLU activation, full data, no weight decay.

#### The Input and Output Phases Lock Together

Each neuron has an input phase $\phi_m$ and an output phase $\psi_m$ (the "shift" of each cosine wave). These are not independent -- training drives them into a precise relationship:

$$\underbrace{\psi_m}_{\text{output phase}} = 2 \times \underbrace{\phi_m}_{\text{input phase}}.$$

**Why "doubled"?** The activation function squares (or, for ReLU, roughly squares) the sum of two cosines. Squaring a cosine at phase $\phi$ naturally produces terms at phase $2\phi$. The output layer learns to match this by setting its own phase to $2\phi$, so the two layers work together coherently.

The **scatter plot** checks this: we plot $2\phi_m$ (horizontal) vs. $\psi_m$ (vertical) for every neuron. If the relationship holds, all points land on the diagonal. This relationship is not built into the architecture -- it **emerges from training** (see Tab 7 for why).

#### Neurons Organize Themselves into a Balanced Ensemble

The neurons don't just specialize to single frequencies -- they also organize *collectively*:

1. **Frequency balance:** Every frequency gets roughly the same number of neurons.
2. **Phase spread:** Within each frequency group, the phases are spread uniformly around the circle. This is what enables **noise cancellation** -- the random noise from individual neurons averages out when their phases are evenly spaced.
3. **Magnitude balance:** All neurons contribute roughly equally to the output (no single neuron dominates).

The **polar plot** shows phases at multiples ($1\times, 2\times, 3\times, 4\times$) on concentric rings -- uniform spread confirms the cancellation condition. The **violin plots** show the distribution of input magnitudes ($\alpha$) and output magnitudes ($\beta$) -- tight concentration confirms magnitude balance.
"""

MATH_TAB4 = r"""
### The Mechanism: Majority Voting in Fourier Space
> **Setup:** ReLU activation, full data, no weight decay.

#### How Neurons Vote for the Correct Answer

Each neuron produces a score for every possible output $j \in \{0, 1, \ldots, p{-}1\}$. Thanks to the phase alignment ($\psi = 2\phi$, see Tab 3), each neuron's score has a **signal** component that peaks at the correct answer $j = (x+y) \bmod p$, plus **noise** that depends on that neuron's particular phase.

When we sum over many neurons within a frequency group, the signal adds up (every neuron agrees on the right answer) while the noise cancels out (different neurons have different phases, and the phase spread from Tab 3 ensures the noise averages to zero). This is **majority voting** -- each neuron casts a noisy vote, but the consensus is correct.

#### The "Flawed Indicator"

After summing over all neurons and all frequency groups, the network's output simplifies to:

$$\text{score for answer } j \;\propto\; \underbrace{\frac{p}{2} \cdot \mathbf{1}[j = (x{+}y) \bmod p]}_{\text{correct answer (strongest)}} \;+\; \underbrace{\frac{p}{4} \cdot \bigl(\mathbf{1}[j = 2x \bmod p] + \mathbf{1}[j = 2y \bmod p]\bigr)}_{\text{two "ghost" peaks (half strength)}}.$$

The correct answer gets score $p/2$, but two **spurious ghost peaks** appear at $2x \bmod p$ and $2y \bmod p$ with score $p/4$. The correct answer always wins because $p/2 > p/4$, so the network always predicts correctly despite the ghosts.

**Heatmap**: The network's output scores for all inputs with $x = 0$. The bright diagonal is the correct answer. The faint lines are the ghost peaks.

**Logit Explorer**: Pick an input pair $(x, y)$ to see the full score distribution. The correct answer (highlighted) should be the tallest bar.
"""

MATH_TAB5 = r"""
### The Lottery Ticket: How Each Neuron Picks Its Frequency
> **Setup:** Quadratic activation ($\sigma(x) = x^2$), full data, random initialization.

#### The Competition

At the start of training, every neuron has a tiny bit of energy at **every** frequency -- nothing is specialized yet. But the input and output phases at each frequency start at random values, so some frequencies happen to be better aligned (input phase and output phase closer to the $\psi = 2\phi$ relationship) than others.

The key insight: **a frequency grows faster when its phases are better aligned.** The growth rate of a frequency's magnitude depends on how close it is to alignment:

$$\text{growth rate} \;\propto\; \cos(\underbrace{2\phi - \psi}_{\text{phase misalignment }\mathcal{D}}).$$

When the misalignment $\mathcal{D}$ is small (phases nearly aligned), $\cos(\mathcal{D}) \approx 1$ and the frequency grows quickly. When $\mathcal{D}$ is large, growth stalls.

#### Winner Takes All

This creates a **positive feedback loop**: the best-aligned frequency grows a little, which helps it align even better, which makes it grow even faster. The gap compounds exponentially until one frequency completely dominates -- **the winner takes all.**

The winning frequency is simply the one that started closest to alignment:

$$\text{winning frequency} = \text{the } k \text{ with smallest initial misalignment } |\mathcal{D}_m^k|.$$

This is a **lottery ticket**: the outcome is determined by the random initialization before training even begins. Since each neuron draws independent random phases, different neurons pick different winning frequencies, naturally producing the balanced frequency coverage seen in Tab 3.

**Phase plot:** Shows how the misalignment $\mathcal{D}$ evolves over training for each frequency within one neuron. The winner (red) converges to zero first; the others barely move.

**Magnitude plot:** Shows how the output magnitude $\beta$ (strength of each frequency) evolves. All start equal. Once the winner aligns, it grows explosively while the others stay frozen.

**Contour plot:** Final magnitude as a function of (initial magnitude, initial misalignment). Largest values appear at small misalignment -- confirming that alignment determines the winner.
"""

MATH_TAB6 = r"""
### Grokking: From Memorization to Generalization
> **Setup:** ReLU activation, 75% training fraction, weight decay $\lambda = 2.0$.

Under the train-test split setup, the network quickly memorizes the training set but takes much longer to generalize. Our analysis reveals grokking is a **three-stage process**, each driven by a different balance of forces.

**Stage I -- Memorization (loss gradient dominates).** The loss gradient dominates and the network rapidly memorizes training data. Training accuracy reaches 100% while test accuracy reaches only ~70%. The ~70% figure (not ~50%) arises because the architecture is symmetric in $x$ and $y$: since $\theta_m[x] + \theta_m[y]$ is invariant under swapping $(x,y) \leftrightarrow (y,x)$, memorizing $(x,y)$ automatically gives the correct answer for $(y,x)$. The lottery mechanism runs on incomplete data, producing a "noisy" multi-frequency representation. We also observe a **common-to-rare ordering**: the network first memorizes symmetric pairs (both $(i,j)$ and $(j,i)$ in training) while actively *suppressing* rare pairs, before eventually memorizing them too.

**Stage II -- Fast Generalization (loss + weight decay).** Weight decay penalizes all magnitudes equally, but the dominant frequency has much larger magnitude and can "afford" the penalty. Non-feature frequencies are driven to zero -- a **sparsification** effect visible as the sharp IPR increase. This transforms the noisy memorization solution into clean single-frequency-per-neuron features. Test accuracy jumps steeply.

**Stage III -- Slow Cleanup (weight decay dominates).** The loss gradient becomes negligible (both losses $\approx 0$). Weight decay alone slowly shrinks norms at rate $\partial_t \|w\| = -\lambda \|w\|$. The feature frequencies are already identified; this stage fine-tunes magnitudes. The network transitions from a lookup table to a generalizing algorithm implementing the indicator function from the mechanism (Tab 4).

**Four progress measures**: (a) Loss -- train drops in Stage I, test drops in Stage II. (b) Accuracy -- train reaches 100% early, test jumps in Stage II. (c) Phase alignment -- $|\sin(\mathcal{D}_m^\star)|$ decreases throughout. (d) IPR + parameter norms -- IPR increases sharply in Stage II, norms shrink in Stage III.

**Epoch Slider**: Use the slider below to see how the accuracy grid evolves across the three stages.
"""

MATH_TAB7 = r"""
### Training Dynamics: Phase Alignment and Single-Frequency Preservation
> **Setup:** Quadratic and ReLU activations, full data, single-frequency initialization, SGD.

#### The Four-Variable ODE

Under small initialization ($\kappa_{\mathrm{init}} \ll 1$), the dynamics decouple: each neuron evolves independently, and within each neuron, different Fourier modes evolve independently (because $\sum_{x \in \mathbb{Z}_p} \cos(\omega_k x) \cos(\omega_\tau x) = \frac{p}{2}\delta_{k,\tau}$). The full dynamics reduce to independent four-variable ODEs per (neuron, frequency):

$$\partial_t \alpha \approx 2p \cdot \alpha \cdot \beta \cdot \cos(\mathcal{D}), \qquad \partial_t \beta \approx p \cdot \alpha^2 \cdot \cos(\mathcal{D}),$$
$$\partial_t \phi \approx 2p \cdot \beta \cdot \sin(\mathcal{D}), \qquad \partial_t \psi \approx -p \cdot \frac{\alpha^2}{\beta} \cdot \sin(\mathcal{D}),$$

where $\mathcal{D} = (2\phi - \psi) \bmod 2\pi$ is the **phase misalignment**. This system has a clear physical interpretation: **magnitudes grow when phases are aligned** ($\cos(\mathcal{D}) \approx 1$), and **phases rotate toward alignment** ($\sin(\mathcal{D}) \to 0$). The dynamics self-coordinate: phases align first (while magnitudes are small), then magnitudes explode.

#### Phase Alignment Theorem

$\mathcal{D}(t) \to 0$ from any initial condition except the measure-zero unstable point $\mathcal{D} = \pi$. The dynamics on the circle behave like an **overdamped pendulum**: $\mathcal{D} = 0$ is a stable attractor, $\mathcal{D} = \pi$ is an unstable repeller. This is not a coincidence or a property of initialization -- it is an **inevitable consequence of the training dynamics**. It explains Observation 2 ($\psi = 2\phi$).

#### Single-Frequency Preservation Theorem

Under the decoupled flow, if a neuron starts at a single frequency, it remains there for all time. The Fourier orthogonality on $\mathbb{Z}_p$ prevents energy from leaking between modes.

**Quadratic** (left panels): Theory matches experiment almost exactly. The DFT heatmap shows the dominant frequency growing while all others stay at zero.

**ReLU** (right panels): Same qualitative behavior with minor quantitative differences. Small energy "leaks" to harmonic multiples ($3k^\star, 5k^\star, \ldots$ for input; $2k^\star, 3k^\star, \ldots$ for output). The leakage decays as $O(r^{-2})$ where $r$ is the harmonic order (third harmonic has $1/9$ the strength, fifth has $1/25$), keeping the dominant frequency overwhelmingly dominant.
"""

MATH_TAB9 = r"""
### Training Log

This tab shows the training logs for each of the 5 configurations run for the selected modulo $p$. Select a run from the dropdown to view its hyperparameters and per-epoch training metrics.

The 5 training runs are:
- **standard**: ReLU, full data, no weight decay -- produces the clean Fourier features analyzed in Tabs 1--5
- **grokking**: ReLU, 75% data, weight decay $\lambda = 2.0$ -- demonstrates the memorization $\to$ generalization transition (Tab 6)
- **quad_random**: Quadratic activation, full data, random init -- the lottery ticket mechanism (Tab 5)
- **quad_single_freq**: Quadratic activation, single-frequency init, SGD -- verifies single-frequency preservation (Tab 7)
- **relu_single_freq**: ReLU, single-frequency init, SGD -- ReLU variant of the dynamics (Tab 7)
"""

MATH_TAB8 = r"""
### Decoupled Gradient Flow Simulation
> **Setup:** Analytical ODE integration (no neural network training).

This tab shows a pure mathematical simulation of the multi-frequency gradient flow, **without training any neural network**. We numerically integrate the four-variable ODEs for all frequency modes simultaneously within a single neuron:

$$\partial_t \alpha_k \approx 2p \cdot \alpha_k \cdot \beta_k \cdot \cos(\mathcal{D}_k), \qquad \partial_t \beta_k \approx p \cdot \alpha_k^2 \cdot \cos(\mathcal{D}_k),$$
$$\partial_t \phi_k \approx 2p \cdot \beta_k \cdot \sin(\mathcal{D}_k), \qquad \partial_t \psi_k \approx -p \cdot \frac{\alpha_k^2}{\beta_k} \cdot \sin(\mathcal{D}_k),$$

for each frequency $k = 1, \ldots, (p{-}1)/2$, with random initial conditions.

The simulation confirms the theoretical predictions from Tab 7:

- **Phase alignment:** Phase misalignments $\mathcal{D}_k = (2\phi_k - \psi_k) \bmod 2\pi$ converge to $0$ for most frequencies, or linger near $\pi$ (the unstable repeller) before eventually escaping.
- **Magnitude competition:** Magnitudes grow explosively for the frequency where $\mathcal{D}_k \approx 0$ first, while others remain near their initial level.
- **Lottery outcome:** The winning frequency (smallest initial $\mathcal{D}_k$) dominates all others, reproducing the full lottery ticket mechanism without any neural network -- just ODEs.

Two cases are shown with different initial conditions to illustrate that the mechanism is robust: regardless of the random starting point, the frequency with the best initial phase alignment always wins.
"""


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

MIN_P = 3  # p=2 has 0 non-DC Fourier frequencies; analysis is degenerate


def get_available_moduli():
    """Discover which p values have pre-computed results (odd p >= 3)."""
    moduli = []
    if os.path.exists(RESULTS_DIR):
        for d in sorted(os.listdir(RESULTS_DIR)):
            if d.startswith("p_"):
                try:
                    p = int(d.split("_")[1])
                    if p >= MIN_P:
                        moduli.append(p)
                except ValueError:
                    pass
    return moduli


def _prime_dir(p):
    return os.path.join(RESULTS_DIR, f"p_{p:03d}")


def load_json_file(p, filename):
    """Load a JSON file from the prime's directory."""
    path = os.path.join(_prime_dir(p), f"p{p:03d}_{filename}")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def safe_img(p, filename):
    """Return image path or None (Gradio handles None gracefully)."""
    path = os.path.join(_prime_dir(p), f"p{p:03d}_{filename}")
    exists = os.path.exists(path)
    if not exists:
        logger.warning(f"Image not found: {path}")
    return path if exists else None


# ---------------------------------------------------------------------------
# Interactive Plotly chart builders
# ---------------------------------------------------------------------------

def _to_np(v):
    """Convert a list/value to a numpy array (bypasses plotly's pandas check)."""
    if v is None:
        return None
    return np.asarray(v)


def make_loss_chart(data, title="Training Loss"):
    """Build an interactive Plotly loss chart from JSON data."""
    if data is None:
        return None
    fig = go.Figure()
    n = len(data.get('train_losses', []))
    epochs = np.arange(n)

    fig.add_trace(go.Scatter(
        x=epochs, y=_to_np(data['train_losses']),
        name='Train Loss', line=dict(color=COLORS[0]),
    ))
    if 'test_losses' in data:
        fig.add_trace(go.Scatter(
            x=epochs, y=_to_np(data['test_losses']),
            name='Test Loss', line=dict(color=COLORS[3]),
        ))

    s1 = data.get('stage1_end')
    s2 = data.get('stage2_end')
    if s1 is not None:
        fig.add_vrect(x0=0, x1=s1, fillcolor=STAGE_COLORS[0],
                      line_width=0, annotation_text="Memorization",
                      annotation_position="top left")
    if s1 is not None and s2 is not None:
        fig.add_vrect(x0=s1, x1=s2, fillcolor=STAGE_COLORS[1],
                      line_width=0, annotation_text="Transition",
                      annotation_position="top left")
    if s2 is not None:
        fig.add_vrect(x0=s2, x1=n, fillcolor=STAGE_COLORS[2],
                      line_width=0, annotation_text="Generalization",
                      annotation_position="top left")

    fig.update_layout(
        title=title, xaxis_title='Epoch', yaxis_title='Loss',
        template='plotly_white', height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    return fig


def make_acc_chart(data, title="Training Accuracy"):
    """Build an interactive Plotly accuracy chart."""
    if data is None:
        return None
    fig = go.Figure()
    epochs = _to_np(data.get('epochs', list(range(len(data.get('train_accs', []))))))

    fig.add_trace(go.Scatter(
        x=epochs, y=_to_np(data['train_accs']),
        name='Train Acc', line=dict(color=COLORS[0]),
    ))
    if 'test_accs' in data:
        fig.add_trace(go.Scatter(
            x=epochs, y=_to_np(data['test_accs']),
            name='Test Acc', line=dict(color=COLORS[3]),
        ))

    s1 = data.get('stage1_end')
    s2 = data.get('stage2_end')
    if s1 is not None:
        fig.add_vrect(x0=0, x1=s1, fillcolor=STAGE_COLORS[0], line_width=0)
    if s1 is not None and s2 is not None:
        fig.add_vrect(x0=s1, x1=s2, fillcolor=STAGE_COLORS[1], line_width=0)
    if s2 is not None:
        n = int(epochs.max()) if len(epochs) > 0 else len(data.get('train_accs', []))
        fig.add_vrect(x0=s2, x1=n, fillcolor=STAGE_COLORS[2], line_width=0)

    fig.update_layout(
        title=title, xaxis_title='Epoch', yaxis_title='Accuracy',
        template='plotly_white', height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    return fig



def make_neuron_spectrum_chart(data, neuron_key):
    """Build a Plotly bar chart for a single neuron's Fourier spectrum."""
    if data is None or neuron_key not in data.get('neurons', {}):
        return None
    neuron = data['neurons'][neuron_key]
    names = data.get('fourier_basis_names', [])
    mags_in = _to_np(neuron['fourier_magnitudes_in'])
    mags_out = _to_np(neuron['fourier_magnitudes_out'])
    dom_freq = neuron.get('dominant_freq', '?')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=mags_in, name='W_in magnitude',
        marker_color=COLORS[0], opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=names, y=mags_out, name='W_out magnitude',
        marker_color=COLORS[3], opacity=0.8,
    ))
    fig.update_layout(
        title=f"Neuron {neuron_key} (dominant freq={dom_freq})",
        xaxis_title='Fourier Component',
        yaxis_title='Magnitude',
        barmode='group',
        template='plotly_white', height=350,
    )
    return fig


def make_logit_bar_chart(data, pair_index):
    """Build a Plotly bar chart of logits for a specific (a,b) pair."""
    if data is None:
        return None
    pairs = data.get('pairs', [])
    logits_all = data.get('logits', [])
    correct = data.get('correct_answers', [])
    classes = data.get('output_classes', [])

    if pair_index >= len(pairs):
        return None

    a, b = pairs[pair_index]
    logits = _to_np(logits_all[pair_index])
    correct_ans = correct[pair_index]

    bar_colors = [COLORS[3] if c == correct_ans else COLORS[0] for c in classes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(c) for c in classes], y=logits,
        marker_color=bar_colors,
        hovertemplate='Class %{x}: %{y:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=f"Logits for ({a}, {b}) -- correct = {correct_ans}",
        xaxis_title='Output Class',
        yaxis_title='Logit Value',
        template='plotly_white', height=350,
    )
    return fig


def make_grokk_heatmap(data, epoch_index):
    """Build a Plotly heatmap of accuracy grid at a grokking checkpoint."""
    if data is None:
        return None
    epochs = data.get('epochs', [])
    grids = data.get('grids', [])
    if epoch_index >= len(grids):
        return None

    grid = _to_np(grids[epoch_index])
    ep = epochs[epoch_index]

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale=[[0, 'white'], [1, COLORS[0]]],
        zmin=0, zmax=1,
        hovertemplate='a=%{y}, b=%{x}: %{z:.0f}<extra></extra>',
    ))
    fig.update_layout(
        title=f"Accuracy Grid at Epoch {ep}",
        xaxis_title='Second Input (b)',
        yaxis_title='First Input (a)',
        template='plotly_white', height=450,
        yaxis=dict(autorange='reversed'),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab update functions
# ---------------------------------------------------------------------------

def update_tab1(p):
    """Overview: standard + grokking loss/IPR, phase scatter."""
    img_overview = safe_img(p, "overview_loss_ipr.png")
    img_phase = safe_img(p, "overview_phase_scatter.png")
    # Also build interactive charts from overview.json
    data = load_json_file(p, "overview.json")
    std_loss_chart = None
    grokk_loss_chart = None
    std_ipr_chart = None
    grokk_ipr_chart = None

    if data:
        # Standard loss chart
        std_ep = data.get('std_epochs', [])
        std_tl = data.get('std_train_loss', [])
        if std_tl:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=_to_np(std_ep[:len(std_tl)]), y=_to_np(std_tl),
                name='Train Loss', line=dict(color=COLORS[0]),
            ))
            fig.update_layout(
                title='Standard: Training Loss (ReLU, full data)',
                xaxis_title='Step', yaxis_title='Loss',
                template='plotly_white', height=350,
            )
            std_loss_chart = fig

        # Standard IPR chart
        std_ipr = data.get('std_ipr', [])
        if std_ipr:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=_to_np(std_ep[:len(std_ipr)]), y=_to_np(std_ipr),
                name='Avg IPR', line=dict(color=COLORS[3]),
            ))
            fig.update_layout(
                title='Standard: IPR (Fourier Sparsity)',
                xaxis_title='Step', yaxis_title='IPR',
                yaxis=dict(range=[0, 1.05]),
                template='plotly_white', height=350,
            )
            std_ipr_chart = fig

        # Grokking loss chart
        grokk_ep = data.get('grokk_epochs', [])
        grokk_tl = data.get('grokk_train_loss', [])
        grokk_tel = data.get('grokk_test_loss', [])
        if grokk_tl or grokk_tel:
            fig = go.Figure()
            if grokk_tl:
                fig.add_trace(go.Scatter(
                    x=_to_np(grokk_ep[:len(grokk_tl)]), y=_to_np(grokk_tl),
                    name='Train Loss', line=dict(color=COLORS[0]),
                ))
            if grokk_tel:
                fig.add_trace(go.Scatter(
                    x=_to_np(grokk_ep[:len(grokk_tel)]), y=_to_np(grokk_tel),
                    name='Test Loss', line=dict(color=COLORS[3]),
                ))
            fig.update_layout(
                title='Grokking: Loss (ReLU, 75% data, WD)',
                xaxis_title='Step', yaxis_title='Loss',
                template='plotly_white', height=350,
            )
            grokk_loss_chart = fig

        # Grokking IPR chart
        grokk_ipr = data.get('grokk_ipr', [])
        if grokk_ipr:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=_to_np(grokk_ep[:len(grokk_ipr)]), y=_to_np(grokk_ipr),
                name='Avg IPR', line=dict(color=COLORS[3]),
            ))
            fig.update_layout(
                title='Grokking: IPR (weight decay drives sparsification)',
                xaxis_title='Step', yaxis_title='IPR',
                yaxis=dict(range=[0, 1.05]),
                template='plotly_white', height=350,
            )
            grokk_ipr_chart = fig

    return (img_overview, std_loss_chart, grokk_loss_chart,
            std_ipr_chart, grokk_ipr_chart, img_phase)


def update_tab2(p):
    """Fourier Weights: heatmap + line plots."""
    return (
        safe_img(p, "full_training_para_origin.png"),
        safe_img(p, "lineplot_in.png"),
        safe_img(p, "lineplot_out.png"),
    )


def update_tab3(p):
    """Phase Analysis: distribution, relationship, magnitude."""
    return (
        safe_img(p, "phase_distribution.png"),
        safe_img(p, "phase_relationship.png"),
        safe_img(p, "magnitude_distribution.png"),
    )


def update_tab4(p):
    """Output Logits."""
    return safe_img(p, "output_logits.png")


def update_tab5(p):
    """Lottery Mechanism: magnitude, phase, contour."""
    return (
        safe_img(p, "lottery_mech_magnitude.png"),
        safe_img(p, "lottery_mech_phase.png"),
        safe_img(p, "lottery_beta_contour.png"),
    )


def update_tab6(p):
    """Grokking: loss/acc charts + analysis images."""
    loss_data = load_json_file(p, "grokk_loss.json")
    acc_data = load_json_file(p, "grokk_acc.json")
    loss_chart = make_loss_chart(loss_data, title="Grokking: Loss")
    acc_chart = make_acc_chart(acc_data, title="Grokking: Accuracy")
    return (
        loss_chart,
        acc_chart,
        safe_img(p, "grokk_abs_phase_diff.png"),
        safe_img(p, "grokk_avg_ipr.png"),
        safe_img(p, "grokk_memorization_accuracy.png"),
        safe_img(p, "grokk_memorization_common_to_rare.png"),
        safe_img(p, "grokk_decoded_weights_dynamic.png"),
    )


def update_tab7(p):
    """Gradient Dynamics: Quad and ReLU single-freq."""
    return (
        safe_img(p, "phase_align_quad.png"),
        safe_img(p, "single_freq_quad.png"),
        safe_img(p, "phase_align_relu.png"),
        safe_img(p, "single_freq_relu.png"),
    )


def update_tab8(p):
    """Decoupled Simulation: 2 analytical gradient flow plots."""
    return (
        safe_img(p, "phase_align_approx1.png"),
        safe_img(p, "phase_align_approx2.png"),
    )


def update_tab9(p):
    """Training Log: return available run names and initial log."""
    data = load_json_file(p, "training_log.json")
    if data is None:
        return [], None, "", ""
    run_names = list(data.keys())
    # Show first run by default
    first_run = run_names[0] if run_names else None
    if first_run:
        run_data = data[first_run]
        config = run_data.get('config', {})
        config_text = _format_config_md(first_run, config)
        log_text = run_data.get('log_text', 'No log available.')
    else:
        config_text = ""
        log_text = ""
    return run_names, first_run, config_text, log_text


def _format_config_md(run_name, config):
    """Format a run config as a Markdown summary."""
    lines = [f"**Run: {run_name}**\n"]
    key_labels = {
        'prime': 'Modulo (p)', 'd_mlp': 'd_mlp',
        'act_type': 'Activation', 'init_type': 'Init Type',
        'init_scale': 'Init Scale', 'optimizer': 'Optimizer',
        'lr': 'Learning Rate', 'weight_decay': 'Weight Decay',
        'frac_train': 'Frac Train', 'num_epochs': 'Num Epochs',
        'seed': 'Seed',
    }
    for key, label in key_labels.items():
        val = config.get(key, 'N/A')
        lines.append(f"- **{label}**: `{val}`")
    return "\n".join(lines)


def update_info(p):
    meta = load_json_file(p, "metadata.json")
    if not meta:
        return f"**p = {p}** | No metadata available"
    d_mlp = meta.get('d_mlp', '?')
    parts = [f"**p = {p}**", f"d_mlp = {d_mlp}"]
    std_metrics = meta.get('final_metrics', {}).get('standard', {})
    if 'train_acc' in std_metrics:
        parts.append(f"Train Acc = {std_metrics['train_acc']:.4f}")
    if 'test_acc' in std_metrics:
        parts.append(f"Test Acc = {std_metrics['test_acc']:.4f}")
    if 'train_loss' in std_metrics:
        parts.append(f"Train Loss = {std_metrics['train_loss']:.6f}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Interactive callback helpers
# ---------------------------------------------------------------------------

def _get_neuron_choices(p):
    """Return list of neuron keys from neuron_spectra.json."""
    data = load_json_file(p, "neuron_spectra.json")
    if data is None:
        return []
    return list(data.get('neurons', {}).keys())


def _get_pair_choices(p):
    """Return list of (a,b) pair labels from logits_interactive.json."""
    data = load_json_file(p, "logits_interactive.json")
    if data is None:
        return []
    pairs = data.get('pairs', [])
    return [f"({a}, {b})" for a, b in pairs]


def _get_grokk_epochs(p):
    """Return list of epoch values from grokk_epoch_data.json."""
    data = load_json_file(p, "grokk_epoch_data.json")
    if data is None:
        return []
    return data.get('epochs', [])


# ---------------------------------------------------------------------------
# Markdown helper -- ensures latex_delimiters are set
# ---------------------------------------------------------------------------

def _md(text, **kwargs):
    """Create a gr.Markdown with KaTeX delimiters enabled."""
    return gr.Markdown(text, latex_delimiters=LATEX_DELIMITERS, **kwargs)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def on_p_change(p_str):
    """Called when the prime dropdown changes. Returns all outputs."""
    p = int(p_str)

    info = update_info(p)

    # Overview
    (t1_img_overview, t1_std_loss, t1_grokk_loss,
     t1_std_ipr, t1_grokk_ipr, t1_phase_scatter) = update_tab1(p)
    # Core Interpretability
    t2_heatmap, t2_line_in, t2_line_out = update_tab2(p)
    t3_phase_dist, t3_phase_rel, t3_magnitude = update_tab3(p)
    t4_logits = update_tab4(p)
    t5_mag, t5_phase, t5_contour = update_tab5(p)
    # Grokking
    (t6_loss, t6_acc, t6_phase_diff, t6_ipr,
     t6_memo, t6_memo_rare, t6_decoded) = update_tab6(p)
    # Theory
    t7_pa_quad, t7_sf_quad, t7_pa_relu, t7_sf_relu = update_tab7(p)
    t8_approx1, t8_approx2 = update_tab8(p)

    # Training Log
    t9_run_names, t9_default_run, t9_config_text, t9_log = update_tab9(p)
    t9_run_dd_update = gr.update(
        choices=t9_run_names,
        value=t9_default_run,
    )

    # Interactive widget updates
    neuron_choices = _get_neuron_choices(p)
    neuron_dd_update = gr.update(
        choices=neuron_choices,
        value=neuron_choices[0] if neuron_choices else None,
    )
    neuron_spectra_data = load_json_file(p, "neuron_spectra.json")
    neuron_chart = make_neuron_spectrum_chart(
        neuron_spectra_data, neuron_choices[0]
    ) if neuron_choices else None

    pair_choices = _get_pair_choices(p)
    pair_dd_update = gr.update(
        choices=pair_choices,
        value=pair_choices[0] if pair_choices else None,
    )
    logit_data = load_json_file(p, "logits_interactive.json")
    logit_chart = make_logit_bar_chart(logit_data, 0) if pair_choices else None

    grokk_epochs = _get_grokk_epochs(p)
    if grokk_epochs:
        slider_update = gr.update(
            minimum=0, maximum=len(grokk_epochs) - 1, value=0, step=1,
            visible=True,
        )
    else:
        slider_update = gr.update(minimum=0, maximum=0, value=0, visible=False)
    grokk_slider_data = load_json_file(p, "grokk_epoch_data.json")
    grokk_heatmap = make_grokk_heatmap(grokk_slider_data, 0) if grokk_epochs else None
    epoch_label = f"Epoch: {grokk_epochs[0]}" if grokk_epochs else ""

    return [
        info,
        # Tab 1: Overview
        t1_img_overview, t1_std_loss, t1_grokk_loss,
        t1_std_ipr, t1_grokk_ipr, t1_phase_scatter,
        # Tab 2: Fourier Weights
        t2_heatmap, t2_line_in, t2_line_out,
        neuron_dd_update, neuron_chart,
        # Tab 3: Phase Analysis
        t3_phase_dist, t3_phase_rel, t3_magnitude,
        # Tab 4: Output Logits
        t4_logits,
        pair_dd_update, logit_chart,
        # Tab 5: Lottery Mechanism
        t5_mag, t5_phase, t5_contour,
        # Tab 6: Grokking
        t6_loss, t6_acc, t6_phase_diff, t6_ipr,
        t6_memo, t6_memo_rare, t6_decoded,
        slider_update, grokk_heatmap, epoch_label,
        # Tab 7: Gradient Dynamics
        t7_pa_quad, t7_sf_quad, t7_pa_relu, t7_sf_relu,
        # Tab 8: Decoupled Simulation
        t8_approx1, t8_approx2,
        # Tab 9: Training Log
        t9_run_dd_update, t9_config_text, t9_log,
    ]


def _commit_results_to_repo(p):
    """Try to commit new precomputed results back to the HF Space repo.

    On HF Spaces, the repo is writable via the huggingface_hub API.
    This allows results to accumulate as users generate them.
    Returns (success, message).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return False, "huggingface_hub not installed"

    space_id = os.environ.get("SPACE_ID")  # e.g. "username/space-name"
    if not space_id:
        return False, "Not running on HF Spaces (SPACE_ID not set)"

    result_dir = os.path.join(RESULTS_DIR, f"p_{p:03d}")
    if not os.path.isdir(result_dir):
        return False, "No results directory found"

    try:
        api = HfApi()
        api.upload_folder(
            folder_path=result_dir,
            path_in_repo=f"precomputed_results/p_{p:03d}",
            repo_id=space_id,
            repo_type="space",
            commit_message=f"Add precomputed results for p={p}",
        )
        return True, f"Committed results for p={p} to {space_id}"
    except Exception as e:
        logger.warning(f"Failed to commit results for p={p}: {e}")
        return False, str(e)


def _run_step_streaming(cmd, env, label):
    """Run a subprocess, yielding (line, error_flag) for each output line."""
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        yield line.rstrip("\n"), False
    proc.wait()
    if proc.returncode != 0:
        yield f"[ERROR] {label} failed (exit code {proc.returncode})", True


def run_pipeline_for_p_streaming(p):
    """Generator: run full pipeline for p, yielding log lines.

    Yields (log_line: str, is_error: bool, is_done: bool).
    Deletes model checkpoints after plot generation to save space.
    """
    if p < 3 or p % 2 == 0:
        yield f"Error: p must be an odd number >= 3, got {p}", True, True
        return
    if p > MAX_P_ON_DEMAND:
        yield f"Error: p={p} exceeds on-demand limit of {MAX_P_ON_DEMAND}", True, True
        return

    result_dir = os.path.join(RESULTS_DIR, f"p_{p:03d}")
    if os.path.isdir(result_dir) and len(os.listdir(result_dir)) > 5:
        yield f"Results for p={p} already exist ({len(os.listdir(result_dir))} files)", False, True
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + ":" + env.get("PYTHONPATH", "")

    steps = [
        ("Step 1/3: Training 5 configurations", [
            sys.executable, "precompute/train_all.py",
            "--p", str(p), "--output", TRAINED_MODELS_DIR, "--resume",
        ]),
        ("Step 2/3: Generating model-based plots", [
            sys.executable, "precompute/generate_plots.py",
            "--p", str(p), "--input", TRAINED_MODELS_DIR,
            "--output", RESULTS_DIR,
        ]),
        ("Step 3/3: Generating analytical plots", [
            sys.executable, "precompute/generate_analytical.py",
            "--p", str(p), "--output", RESULTS_DIR,
        ]),
    ]

    for label, cmd in steps:
        yield f"\n{'='*60}", False, False
        yield f"  {label} (p={p})", False, False
        yield f"{'='*60}", False, False
        for line, is_err in _run_step_streaming(cmd, env, label):
            if is_err:
                yield line, True, True
                return
            yield line, False, False

    # Cleanup checkpoints
    model_dir = os.path.join(TRAINED_MODELS_DIR, f"p_{p:03d}")
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
        yield f"Cleaned up checkpoints: {model_dir}", False, False

    n_files = len(os.listdir(result_dir)) if os.path.isdir(result_dir) else 0

    # Try to commit results back to the HF repo
    ok_commit, commit_msg = _commit_results_to_repo(p)
    if ok_commit:
        yield f"Results saved to HF repo.", False, False

    yield f"\nDone! Generated {n_files} files for p={p}.", False, True


def create_app():
    moduli = get_available_moduli()
    p_choices = [str(p) for p in moduli]
    default_p = p_choices[0] if p_choices else None

    with gr.Blocks(
        title="Modular Addition Feature Learning",
    ) as app:
        _md(
            r"# On the Mechanism and Dynamics of Modular Addition" "\n"
            r"### Fourier Features, Lottery Ticket, and Grokking" "\n\n"
            r"**Jianliang He, Leda Wang, Siyu Chen, Zhuoran Yang**" "\n"
            r"*Department of Statistics and Data Science, Yale University*" "\n\n"
            r'<a href="#">[arXiv]</a> &nbsp; '
            r'<a href="#">[Blog]</a> &nbsp; '
            r'<a href="https://github.com/Y-Agent/modular-addition-feature-learning">[Code]</a>' "\n\n"
            r"---" "\n\n"
            r"This interactive explorer visualizes how a two-layer neural network "
            r"learns modular arithmetic $f(x,y) = (x + y) \bmod p$ through "
            r"Fourier feature learning, lottery ticket dynamics, and the grokking "
            r"phenomenon. Select a modulo $p$ (any odd number $\geq 3$) below to view pre-computed results." "\n\n"
            r"> **Note:** Grokking experiments (Tab 6) require $p \geq 19$ to have enough data for a meaningful train/test split. "
            r"For $p < 19$, grokking plots will not be generated."
        )

        # Hidden state for current modulo
        current_p = gr.State(value=int(default_p) if default_p else 3)

        with gr.Row():
            p_dropdown = gr.Dropdown(
                choices=p_choices,
                value=default_p,
                label="Select Modulo (p)",
                interactive=True,
                scale=2,
            )
            info_md = _md(
                update_info(int(default_p)) if default_p else ""
            )

        with gr.Accordion("Generate results for a new p", open=False):
            _md(
                f"Enter any odd number $p \\geq 3$ (max {MAX_P_ON_DEMAND} "
                f"for on-demand training). This will train 5 model "
                f"configurations and generate all plots. Training logs "
                f"are streamed below in real time."
            )
            with gr.Row():
                new_p_input = gr.Number(
                    value=None, label="New p (odd, ≥ 3)",
                    precision=0, scale=1,
                )
                generate_btn = gr.Button(
                    "Generate", variant="primary", scale=1,
                )
            generate_status = _md("")
            generate_log = gr.Code(
                value="", language=None, label="Pipeline Log",
                lines=15, interactive=False, visible=False,
            )

        # ----- Tabs -----
        with gr.Tabs():

            # === Core Interpretability ===

            # Tab 1: Overview
            with gr.Tab("1. Overview"):
                _md(MATH_TAB1)
                t1_img_overview = gr.Image(
                    label="Loss & IPR Overview (Static)", type="filepath"
                )
                with gr.Row():
                    t1_std_loss = gr.Plot(label="Standard: Loss")
                    t1_grokk_loss = gr.Plot(label="Grokking: Loss")
                with gr.Row():
                    t1_std_ipr = gr.Plot(label="Standard: IPR")
                    t1_grokk_ipr = gr.Plot(label="Grokking: IPR")
                t1_phase_scatter = gr.Image(
                    label="Phase Alignment: \u03c8 = 2\u03c6", type="filepath"
                )

            # Tab 2: Fourier Weights
            with gr.Tab("2. Fourier Weights"):
                _md(MATH_TAB2)
                t2_heatmap = gr.Image(label="Decoded W_in / W_out Heatmap", type="filepath")
                with gr.Row():
                    t2_line_in = gr.Image(label="First-Layer Line Plots (with cosine fit)", type="filepath")
                    t2_line_out = gr.Image(label="Second-Layer Line Plots (with cosine fit)", type="filepath")
                _md("#### Neuron Frequency Inspector")
                t2_neuron_dd = gr.Dropdown(
                    choices=[], value=None,
                    label="Select Neuron", interactive=True,
                )
                t2_neuron_chart = gr.Plot(label="Neuron Fourier Spectrum")

            # Tab 3: Phase Analysis
            with gr.Tab("3. Phase Analysis"):
                _md(MATH_TAB3)
                with gr.Row():
                    t3_phase_dist = gr.Image(label="Phase Distribution", type="filepath")
                    t3_phase_rel = gr.Image(
                        label="Phase Relationship (2\u03c6 vs \u03c8)", type="filepath"
                    )
                t3_magnitude = gr.Image(label="Magnitude Distribution", type="filepath")

            # Tab 4: Output Logits
            with gr.Tab("4. Output Logits"):
                _md(MATH_TAB4)
                t4_logits = gr.Image(label="Output Logits Heatmap", type="filepath")
                _md("#### Logit Explorer")
                t4_pair_dd = gr.Dropdown(
                    choices=[], value=None,
                    label="Select Input Pair (a, b)", interactive=True,
                )
                t4_logit_chart = gr.Plot(label="Logit Distribution")

            # Tab 5: Lottery Mechanism
            with gr.Tab("5. Lottery Mechanism"):
                _md(MATH_TAB5)
                _md(r"""**Magnitude plot** below: Each curve tracks one frequency's output magnitude $\beta_k$ within a single neuron over training. All frequencies start with equal magnitude (from random initialization). The winning frequency (best initial phase alignment) grows explosively while others remain frozen.""")
                t5_mag = gr.Image(label="Frequency Magnitude Evolution", type="filepath")
                _md(r"""**Phase plot** below: Each curve shows the phase misalignment $\mathcal{D}_k = 2\phi_k - \psi_k$ for one frequency within the same neuron. The winning frequency (colored) converges to $\mathcal{D} = 0$ (perfect alignment) first; other frequencies barely change because their magnitudes remain small.""")
                t5_phase = gr.Image(label="Phase Misalignment Convergence", type="filepath")
                _md(r"""**Contour plot** below: Final output magnitude as a function of initial magnitude and initial phase misalignment, across all neurons. The largest final magnitudes (brightest regions) appear at small initial misalignment $|\mathcal{D}|$, confirming that initial phase alignment -- not initial magnitude -- determines which frequency wins.""")
                t5_contour = gr.Image(label="Final Magnitude Contour", type="filepath")

            # === Grokking ===

            # Tab 6: Grokking
            with gr.Tab("6. Grokking"):
                _md(MATH_TAB6)

                _md(r"""#### (a) Loss and (b) Accuracy

**(a) Loss:** Training loss (blue) drops rapidly in Stage I as the network memorizes training data. Test loss (red) stays high until Stage II, when weight decay forces the network to find a generalizing solution, causing test loss to plummet. The three colored bands mark the three stages.

**(b) Accuracy:** Training accuracy reaches 100% early (Stage I). Test accuracy stays at ~70% during memorization (not 50% -- the built-in symmetry $f(a,b) = f(b,a)$ gives "free" correct answers for the swapped pair). Test accuracy jumps sharply in Stage II when the network transitions from memorization to Fourier features.""")
                with gr.Row():
                    t6_loss = gr.Plot(label="Grokking Loss (Interactive)")
                    t6_acc = gr.Plot(label="Grokking Accuracy (Interactive)")

                _md(r"""#### (c) Phase Alignment and (d) IPR & Norms

**(c) Phase alignment:** Average $|\sin(\mathcal{D}_m^\star)|$ over all neurons, where $\mathcal{D}_m^\star = 2\phi_m^\star - \psi_m^\star$ is the phase misalignment at each neuron's dominant frequency. This measures how far the network is from the ideal relationship $\psi = 2\phi$. It decreases throughout training as phases align, with the steepest drop during Stage II.

**(d) IPR and parameter norms:** IPR (Fourier sparsity) increases sharply in Stage II -- this is the "aha" moment where multi-frequency noise collapses into clean single-frequency features. Parameter norms shrink steadily in Stage III as weight decay slowly polishes the solution.""")
                with gr.Row():
                    t6_phase_diff = gr.Image(
                        label="Phase Difference |sin(D*)|", type="filepath"
                    )
                    t6_ipr = gr.Image(label="IPR & Parameter Norms", type="filepath")

                _md(r"""#### (e) Memorization Accuracy Grid

Each cell $(i,j)$ in the grid shows whether the network correctly predicts $(i+j) \bmod p$ at a given training epoch. **White = correct, dark = incorrect.** Training pairs are marked with dots.

During Stage I, the network first memorizes **symmetric pairs** -- pairs where both $(i,j)$ and $(j,i)$ are in the training set (they appear on both sides of the diagonal). These are learned first because the architecture treats inputs symmetrically: $\theta_m[i] + \theta_m[j] = \theta_m[j] + \theta_m[i]$, so learning one automatically gives the other.

**Asymmetric pairs** (where only one of $(i,j)$ or $(j,i)$ is in training) are harder to memorize and are learned later. Some test pairs may even be *actively suppressed* (the network gets them wrong on purpose) before eventually being memorized.""")
                t6_memo = gr.Image(label="Memorization Accuracy", type="filepath")

                _md(r"""#### (f) Common-to-Rare Ordering

This plot reorders the accuracy grid to reveal the **memorization sequence**. Instead of plotting by input value, it sorts pairs by how "common" they are in the training set:

- **Common pairs** (top-left): Both $(i,j)$ and $(j,i)$ in training set. These are memorized first.
- **Rare pairs** (bottom-right): Only one ordering in training set. These are memorized last, and may be temporarily suppressed before being learned.

The plot shows a clear **top-left to bottom-right** progression, confirming that the network memorizes common pairs before rare ones.""")
                t6_memo_rare = gr.Image(label="Memorization: Common to Rare", type="filepath")

                _md(r"""#### (g) Decoded Weights Across Stages

DFT heatmaps of the network's weights at key epochs through the three stages. Each row is a neuron; each column is a Fourier frequency component.

- **Stage I (memorization):** Weights are noisy with energy spread across many frequencies -- the network is using all available capacity to memorize.
- **Stage II (generalization):** Weight decay kills the weak frequencies. Each neuron's energy concentrates into a single frequency -- clean Fourier features emerge.
- **Stage III (cleanup):** Features are already clean; weight decay slowly shrinks overall magnitude without changing the structure.""")
                t6_decoded = gr.Image(label="Decoded Weights Across Stages", type="filepath")

                _md(r"""#### Accuracy Grid Across Training (Interactive)

Use the slider to scrub through training epochs and watch the accuracy grid evolve. In Stage I, you'll see the symmetric pairs (along both diagonals) light up first, then asymmetric pairs fill in, and finally the entire grid becomes white in Stage II as the network generalizes.""")
                t6_slider = gr.Slider(
                    minimum=0, maximum=0, value=0, step=1,
                    label="Epoch Snapshot Index", interactive=True,
                    visible=False,
                )
                t6_heatmap = gr.Plot(label="Accuracy Heatmap")
                t6_epoch_label = _md("")

            # === Theory ===

            # Tab 7: Gradient Dynamics
            with gr.Tab("7. Gradient Dynamics"):
                _md(MATH_TAB7)
                _md(r"""#### Quadratic Activation ($\sigma(x) = x^2$)

**Left -- Phase alignment:** Tracks the input phase $\phi_m^\star$, output phase $\psi_m^\star$, and doubled input phase $2\phi_m^\star$ of the dominant frequency in a single neuron over training. The theory predicts $\psi \to 2\phi$; here we see $\psi$ (red) and $2\phi$ (blue) converge and overlap, confirming phase alignment. The phases lock in early while magnitudes are still small.

**Right -- DFT heatmaps:** Decoded weights in Fourier space at key training steps. At step 0, the neuron starts with energy at a single frequency (by construction -- single-frequency initialization). At later steps, the dominant frequency grows while all other frequencies stay at zero. This confirms the **single-frequency preservation theorem**: Fourier orthogonality prevents energy leakage between modes.""")
                with gr.Row():
                    t7_pa_quad = gr.Image(label="Phase Alignment (Quad)", type="filepath")
                    t7_sf_quad = gr.Image(label="Decoded Weights (Quad)", type="filepath")
                _md(r"""#### ReLU Activation ($\sigma(x) = \max(0, x)$)

**Left -- Phase alignment:** Same as quadratic above, but with ReLU. The qualitative behavior is identical: $\psi$ converges to $2\phi$. Minor quantitative differences arise because ReLU is not exactly $x^2$.

**Right -- DFT heatmaps:** Unlike quadratic, ReLU leaks small amounts of energy to **harmonic multiples** of the dominant frequency ($3k^\star, 5k^\star, \ldots$ for input weights; $2k^\star, 3k^\star, \ldots$ for output weights). This leakage decays as $O(r^{-2})$ where $r$ is the harmonic order, so the dominant frequency remains overwhelmingly dominant. The faint "stripes" at harmonic positions are this leakage.""")
                with gr.Row():
                    t7_pa_relu = gr.Image(label="Phase Alignment (ReLU)", type="filepath")
                    t7_sf_relu = gr.Image(label="Decoded Weights (ReLU)", type="filepath")

            # Tab 8: Decoupled Simulation
            with gr.Tab("8. Decoupled Simulation"):
                _md(MATH_TAB8)
                _md(r"""Each 3-panel figure below shows one simulation run. The gray curves are non-winning frequencies; the colored curves are the winning frequency $k^\star$.

**Top panel -- Phase alignment:** $\psi_{k^\star}$ (red) and $2\phi_{k^\star}$ (blue) converge toward each other, confirming that training drives phases into the $\psi = 2\phi$ relationship even in this pure ODE setting (no neural network).

**Middle panel -- Phase difference $D_{k^\star}$:** The misalignment $\mathcal{D}_{k^\star} = 2\phi_{k^\star} - \psi_{k^\star}$ converges toward $0$ (or $\pi/2$ transiently in Case 1). The dashed horizontal line marks $\pi/2$. Non-winning frequencies (gray) remain scattered because their magnitudes are too small to drive phase alignment.

**Bottom panel -- Magnitude evolution:** The winning frequency's magnitudes ($\alpha_{k^\star}$ and $\beta_{k^\star}$) grow explosively once phase alignment is achieved, while all other frequencies remain near their initial values. This is the lottery ticket effect in pure form.""")
                with gr.Row():
                    t8_approx1 = gr.Image(
                        label="Gradient Flow (Case 1: with annotations)", type="filepath"
                    )
                    t8_approx2 = gr.Image(label="Gradient Flow (Case 2)", type="filepath")

            # Tab 9: Training Log
            with gr.Tab("9. Training Log"):
                _md(MATH_TAB9)
                t9_run_dd = gr.Dropdown(
                    choices=[], value=None,
                    label="Select Training Run", interactive=True,
                )
                t9_config_md = _md("")
                t9_log_text = gr.Code(
                    value="", language=None, label="Training Log",
                    lines=30, interactive=False,
                )

        # All outputs for prime change
        all_outputs = [
            info_md,
            # Tab 1: Overview
            t1_img_overview, t1_std_loss, t1_grokk_loss,
            t1_std_ipr, t1_grokk_ipr, t1_phase_scatter,
            # Tab 2
            t2_heatmap, t2_line_in, t2_line_out,
            t2_neuron_dd, t2_neuron_chart,
            # Tab 3
            t3_phase_dist, t3_phase_rel, t3_magnitude,
            # Tab 4
            t4_logits,
            t4_pair_dd, t4_logit_chart,
            # Tab 5
            t5_mag, t5_phase, t5_contour,
            # Tab 6
            t6_loss, t6_acc, t6_phase_diff, t6_ipr,
            t6_memo, t6_memo_rare, t6_decoded,
            t6_slider, t6_heatmap, t6_epoch_label,
            # Tab 7
            t7_pa_quad, t7_sf_quad, t7_pa_relu, t7_sf_relu,
            # Tab 8
            t8_approx1, t8_approx2,
            # Tab 9
            t9_run_dd, t9_config_md, t9_log_text,
        ]

        # --- Prime change handler ---
        def p_change_and_store(p_str):
            p = int(p_str)
            results = on_p_change(p_str)
            return [p] + results

        p_dropdown.change(
            fn=p_change_and_store,
            inputs=[p_dropdown],
            outputs=[current_p] + all_outputs,
        )

        # Trigger initial load so tabs aren't empty on page load
        app.load(
            fn=p_change_and_store,
            inputs=[p_dropdown],
            outputs=[current_p] + all_outputs,
        )

        # --- Neuron dropdown handler ---
        def on_neuron_change(neuron_key, p):
            data = load_json_file(p, "neuron_spectra.json")
            return make_neuron_spectrum_chart(data, neuron_key)

        t2_neuron_dd.change(
            fn=on_neuron_change,
            inputs=[t2_neuron_dd, current_p],
            outputs=[t2_neuron_chart],
        )

        # --- Logit pair dropdown handler ---
        def on_pair_change(pair_str, p):
            data = load_json_file(p, "logits_interactive.json")
            if data is None or not pair_str:
                return None
            pairs = data.get('pairs', [])
            pair_labels = [f"({a}, {b})" for a, b in pairs]
            if pair_str in pair_labels:
                idx = pair_labels.index(pair_str)
            else:
                idx = 0
            return make_logit_bar_chart(data, idx)

        t4_pair_dd.change(
            fn=on_pair_change,
            inputs=[t4_pair_dd, current_p],
            outputs=[t4_logit_chart],
        )

        # --- Grokking slider handler ---
        def on_grokk_slider(slider_val, p):
            data = load_json_file(p, "grokk_epoch_data.json")
            if data is None:
                return None, ""
            idx = int(slider_val)
            epochs = data.get('epochs', [])
            label = f"**Epoch: {epochs[idx]}**" if idx < len(epochs) else ""
            return make_grokk_heatmap(data, idx), label

        t6_slider.change(
            fn=on_grokk_slider,
            inputs=[t6_slider, current_p],
            outputs=[t6_heatmap, t6_epoch_label],
        )

        # --- Training log run dropdown handler ---
        def on_log_run_change(run_name, p):
            data = load_json_file(p, "training_log.json")
            if data is None or run_name not in data:
                return "", ""
            run_data = data[run_name]
            config = run_data.get('config', {})
            config_text = _format_config_md(run_name, config)
            log_text = run_data.get('log_text', 'No log available.')
            return config_text, log_text

        t9_run_dd.change(
            fn=on_log_run_change,
            inputs=[t9_run_dd, current_p],
            outputs=[t9_config_md, t9_log_text],
        )

        # --- On-demand training handler (streaming) ---
        def on_generate_click(new_p):
            if new_p is None:
                yield (
                    gr.update(), gr.update(),
                    "Enter a value for p.",
                    gr.update(visible=False, value=""),
                )
                return
            p = int(new_p)
            log_lines = []
            yield (
                gr.update(), gr.update(),
                f"**Running pipeline for p={p}...**",
                gr.update(visible=True, value="Starting...\n"),
            )
            for line, is_err, is_done in run_pipeline_for_p_streaming(p):
                log_lines.append(line)
                # Keep last 200 lines to avoid memory bloat
                display = "\n".join(log_lines[-200:])
                if is_err:
                    yield (
                        gr.update(), gr.update(),
                        f"**Error:** {line}",
                        gr.update(value=display),
                    )
                    return
                if is_done:
                    new_moduli = get_available_moduli()
                    new_choices = [str(v) for v in new_moduli]
                    yield (
                        gr.update(choices=new_choices, value=str(p)),
                        gr.update(),
                        f"**Success:** {line}",
                        gr.update(value=display),
                    )
                    return
                yield (
                    gr.update(), gr.update(),
                    f"**Running pipeline for p={p}...**",
                    gr.update(value=display),
                )

        generate_btn.click(
            fn=on_generate_click,
            inputs=[new_p_input],
            outputs=[p_dropdown, current_p, generate_status, generate_log],
        )

    return app


if __name__ == "__main__":
    # Startup diagnostics
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"RESULTS_DIR:  {RESULTS_DIR}")
    print(f"RESULTS_DIR exists: {os.path.exists(RESULTS_DIR)}")
    if os.path.exists(RESULTS_DIR):
        dirs = sorted(os.listdir(RESULTS_DIR))
        print(f"Result dirs: {dirs}")
        for d in dirs[:2]:
            dpath = os.path.join(RESULTS_DIR, d)
            files = os.listdir(dpath) if os.path.isdir(dpath) else []
            print(f"  {d}: {len(files)} files")
            for f in sorted(files)[:5]:
                print(f"    {f}")
    else:
        print("WARNING: RESULTS_DIR does not exist!")

    app = create_app()
    app.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS, ssr_mode=False)
