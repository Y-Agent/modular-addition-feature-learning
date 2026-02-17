#!/usr/bin/env python3
"""
Gradio app for Modular Addition Feature Learning visualization.
Serves pre-computed results for primes 3-199.

All results are pre-computed as PNG images and JSON data files.
No GPU needed at serving time.
"""
import gradio as gr
import json
import os
import plotly.graph_objects as go
from pathlib import Path

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "precomputed_results")

COLORS = ['#0D2758', '#60656F', '#DEA54B', '#A32015', '#347186']
STAGE_COLORS = ['rgba(212,175,55,0.15)', 'rgba(139,115,85,0.15)', 'rgba(192,192,192,0.15)']


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def get_available_primes():
    """Discover which primes have pre-computed results."""
    primes = []
    if os.path.exists(RESULTS_DIR):
        for d in sorted(os.listdir(RESULTS_DIR)):
            if d.startswith("p_"):
                try:
                    p = int(d.split("_")[1])
                    primes.append(p)
                except ValueError:
                    pass
    return primes


def load_metadata(p):
    path = os.path.join(RESULTS_DIR, f"p_{p:03d}", "metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def img_path(p, tab_dir, filename):
    path = os.path.join(RESULTS_DIR, f"p_{p:03d}", tab_dir, filename)
    return path if os.path.exists(path) else None


def load_json(p, tab_dir, filename):
    path = os.path.join(RESULTS_DIR, f"p_{p:03d}", tab_dir, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


NOT_COMPUTED = os.path.join(os.path.dirname(__file__), "not_computed.png")


def safe_img(p, tab_dir, filename):
    """Return image path or None (Gradio handles None gracefully)."""
    return img_path(p, tab_dir, filename)


# ---------------------------------------------------------------------------
# Interactive Plotly chart builders
# ---------------------------------------------------------------------------

def make_loss_chart(data, title="Training Loss"):
    """Build an interactive Plotly loss chart from JSON data."""
    if data is None:
        return None
    fig = go.Figure()
    n = len(data.get('train_losses', []))
    epochs = list(range(n))

    fig.add_trace(go.Scatter(
        x=epochs, y=data['train_losses'],
        name='Train Loss', line=dict(color=COLORS[0]),
    ))
    if 'test_losses' in data:
        fig.add_trace(go.Scatter(
            x=epochs, y=data['test_losses'],
            name='Test Loss', line=dict(color=COLORS[3]),
        ))

    # Stage shading for grokking
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
    epochs = data.get('epochs', list(range(len(data.get('train_accs', [])))))

    fig.add_trace(go.Scatter(
        x=epochs, y=data['train_accs'],
        name='Train Acc', line=dict(color=COLORS[0]),
    ))
    if 'test_accs' in data:
        fig.add_trace(go.Scatter(
            x=epochs, y=data['test_accs'],
            name='Test Acc', line=dict(color=COLORS[3]),
        ))

    s1 = data.get('stage1_end')
    s2 = data.get('stage2_end')
    if s1 is not None:
        fig.add_vrect(x0=0, x1=s1, fillcolor=STAGE_COLORS[0], line_width=0)
    if s1 is not None and s2 is not None:
        fig.add_vrect(x0=s1, x1=s2, fillcolor=STAGE_COLORS[1], line_width=0)
    if s2 is not None:
        n = max(epochs) if epochs else len(data.get('train_accs', []))
        fig.add_vrect(x0=s2, x1=n, fillcolor=STAGE_COLORS[2], line_width=0)

    fig.update_layout(
        title=title, xaxis_title='Epoch', yaxis_title='Accuracy',
        template='plotly_white', height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    return fig


def make_ipr_chart(data, title="Loss & Sparsity (IPR)"):
    """Build Plotly chart with loss and IPR on separate y-axes."""
    if data is None:
        return None
    fig = go.Figure()

    # Loss trace
    n_loss = len(data.get('train_losses', []))
    fig.add_trace(go.Scatter(
        x=list(range(n_loss)), y=data['train_losses'],
        name='Train Loss', line=dict(color=COLORS[0]),
    ))

    # IPR trace on secondary y-axis
    ipr_epochs = data.get('ipr_epochs', [])
    ipr_values = data.get('ipr_values', [])
    if ipr_epochs and ipr_values:
        fig.add_trace(go.Scatter(
            x=ipr_epochs, y=ipr_values,
            name='Avg IPR', line=dict(color=COLORS[3]),
            yaxis='y2',
        ))

    fig.update_layout(
        title=title, xaxis_title='Epoch',
        yaxis=dict(title='Loss', titlefont=dict(color=COLORS[0])),
        yaxis2=dict(title='IPR', titlefont=dict(color=COLORS[3]),
                    overlaying='y', side='right'),
        template='plotly_white', height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab update functions
# ---------------------------------------------------------------------------

def update_tab1(p):
    data = load_json(p, "tab1_training_overview", "loss_sparsity.json")
    chart = make_ipr_chart(data)
    img = safe_img(p, "tab1_training_overview", "loss_sparsity.png")
    return chart, img


def update_tab2(p):
    return (
        safe_img(p, "tab2_fourier_weights", "full_training_para_origin.png"),
        safe_img(p, "tab2_fourier_weights", "full_training_para_origin_lineplot_in.png"),
        safe_img(p, "tab2_fourier_weights", "full_training_para_origin_lineplot_out.png"),
    )


def update_tab3(p):
    return (
        safe_img(p, "tab3_phase_analysis", "full_training_phase_distribution.png"),
        safe_img(p, "tab3_phase_analysis", "full_training_phase_relationship.png"),
        safe_img(p, "tab3_phase_analysis", "full_training_magnitude_distribution.png"),
    )


def update_tab4(p):
    return safe_img(p, "tab4_output_logits", "output_logits.png")


def update_tab5(p):
    loss_data = load_json(p, "tab5_grokking", "grokk_loss.json")
    acc_data = load_json(p, "tab5_grokking", "grokk_acc.json")
    loss_chart = make_loss_chart(loss_data, title="Grokking: Loss")
    acc_chart = make_acc_chart(acc_data, title="Grokking: Accuracy")
    return (
        loss_chart,
        acc_chart,
        safe_img(p, "tab5_grokking", "grokk_abs_phase_diff.png"),
        safe_img(p, "tab5_grokking", "grokk_avg_ipr.png"),
        safe_img(p, "tab5_grokking", "grokk_memorization_accuracy.png"),
        safe_img(p, "tab5_grokking", "grokk_memorization_common_to_rare.png"),
        safe_img(p, "tab5_grokking", "grokk_decoded_weights_dynamic.png"),
    )


def update_tab6(p):
    return (
        safe_img(p, "tab6_lottery", "lottery_mech_magnitude.png"),
        safe_img(p, "tab6_lottery", "lottery_mech_phase.png"),
        safe_img(p, "tab6_lottery", "lottery_beta_contour.png"),
    )


def update_tab7(p):
    return (
        safe_img(p, "tab7_gradient_dynamics", "phase_align_quad.png"),
        safe_img(p, "tab7_gradient_dynamics", "single_freq_quad.png"),
        safe_img(p, "tab7_gradient_dynamics", "phase_align_relu.png"),
        safe_img(p, "tab7_gradient_dynamics", "single_freq_relu.png"),
    )


def update_tab8(p):
    return (
        safe_img(p, "tab8_theory", "phase_align_approx1.png"),
        safe_img(p, "tab8_theory", "phase_align_approx2.png"),
        safe_img(p, "tab8_theory", "frequency_diversity_output_logits.png"),
        safe_img(p, "tab8_theory", "phase_diversity_output_logits.png"),
    )


def update_info(p):
    meta = load_metadata(p)
    if not meta:
        return f"**Prime p = {p}** | No metadata available"
    d_mlp = meta.get('d_mlp', '?')
    train_acc = meta.get('final_train_acc')
    test_acc = meta.get('final_test_acc')
    train_loss = meta.get('final_train_loss')
    parts = [f"**Prime p = {p}**", f"d_mlp = {d_mlp}"]
    if train_acc is not None:
        parts.append(f"Train Acc = {train_acc:.4f}")
    if test_acc is not None:
        parts.append(f"Test Acc = {test_acc:.4f}")
    if train_loss is not None:
        parts.append(f"Train Loss = {train_loss:.6f}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def on_prime_change(prime_str):
    """Called when the prime dropdown changes. Returns all outputs."""
    p = int(prime_str)

    info = update_info(p)

    # Tab 1
    t1_chart, t1_img = update_tab1(p)
    # Tab 2
    t2_heatmap, t2_line_in, t2_line_out = update_tab2(p)
    # Tab 3
    t3_phase_dist, t3_phase_rel, t3_magnitude = update_tab3(p)
    # Tab 4
    t4_logits = update_tab4(p)
    # Tab 5
    (t5_loss, t5_acc, t5_phase_diff, t5_ipr,
     t5_memo, t5_memo_rare, t5_decoded) = update_tab5(p)
    # Tab 6
    t6_mag, t6_phase, t6_contour = update_tab6(p)
    # Tab 7
    t7_pa_quad, t7_sf_quad, t7_pa_relu, t7_sf_relu = update_tab7(p)
    # Tab 8
    t8_approx1, t8_approx2, t8_freq_div, t8_phase_div = update_tab8(p)

    return [
        info,
        # Tab 1
        t1_chart, t1_img,
        # Tab 2
        t2_heatmap, t2_line_in, t2_line_out,
        # Tab 3
        t3_phase_dist, t3_phase_rel, t3_magnitude,
        # Tab 4
        t4_logits,
        # Tab 5
        t5_loss, t5_acc, t5_phase_diff, t5_ipr,
        t5_memo, t5_memo_rare, t5_decoded,
        # Tab 6
        t6_mag, t6_phase, t6_contour,
        # Tab 7
        t7_pa_quad, t7_sf_quad, t7_pa_relu, t7_sf_relu,
        # Tab 8
        t8_approx1, t8_approx2, t8_freq_div, t8_phase_div,
    ]


def create_app():
    primes = get_available_primes()
    default_prime = str(primes[0]) if primes else "3"

    with gr.Blocks(
        title="Modular Addition Feature Learning",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Modular Addition Feature Learning Explorer\n"
            "Explore how neural networks learn modular arithmetic (a + b mod p) "
            "through Fourier feature learning, grokking, and lottery ticket phenomena.\n\n"
            "Select a prime below to view pre-computed results."
        )

        with gr.Row():
            prime_dropdown = gr.Dropdown(
                choices=[str(p) for p in primes],
                value=default_prime,
                label="Select Prime (p)",
                interactive=True,
                scale=1,
            )
            info_md = gr.Markdown(value=update_info(int(default_prime)))

        # ----- Tabs -----
        with gr.Tabs():

            # Tab 1: Training Overview
            with gr.Tab("Training Overview"):
                gr.Markdown("### Loss & Sparsity (IPR) During Training")
                gr.Markdown("Interactive chart: hover to see values, zoom to inspect regions.")
                t1_chart = gr.Plot(label="Loss & IPR (Interactive)")
                t1_img = gr.Image(label="Loss & Sparsity (Static)", type="filepath")

            # Tab 2: Fourier Weights
            with gr.Tab("Fourier Weights"):
                gr.Markdown(
                    "### Decoded Weights in Fourier Basis\n"
                    "Top neurons sorted by dominant frequency. "
                    "Each row is a neuron, each column is a Fourier component."
                )
                t2_heatmap = gr.Image(label="Decoded W_in / W_out Heatmap", type="filepath")
                with gr.Row():
                    t2_line_in = gr.Image(label="First-Layer Line Plots (with cosine fit)", type="filepath")
                    t2_line_out = gr.Image(label="Second-Layer Line Plots (with cosine fit)", type="filepath")

            # Tab 3: Phase Analysis
            with gr.Tab("Phase Analysis"):
                gr.Markdown(
                    "### Phase & Magnitude Analysis\n"
                    "Phase distribution on concentric circles, "
                    "phase alignment (2\u03c6 vs \u03c8), "
                    "and input/output magnitude distributions."
                )
                with gr.Row():
                    t3_phase_dist = gr.Image(label="Phase Distribution", type="filepath")
                    t3_phase_rel = gr.Image(label="Phase Relationship (2\u03c6 vs \u03c8)", type="filepath")
                t3_magnitude = gr.Image(label="Magnitude Distribution", type="filepath")

            # Tab 4: Output Logits
            with gr.Tab("Output Logits"):
                gr.Markdown(
                    "### Model Output Logits\n"
                    "Heatmap of logits for input pairs. "
                    "Rectangles mark the correct answer (x+y mod p)."
                )
                t4_logits = gr.Image(label="Output Logits Heatmap", type="filepath")

            # Tab 5: Grokking
            with gr.Tab("Grokking"):
                gr.Markdown(
                    "### Grokking Phenomenon\n"
                    "Training with weight decay and partial data (75% train). "
                    "Three stages: memorization \u2192 transition \u2192 generalization."
                )
                with gr.Row():
                    t5_loss = gr.Plot(label="Grokking Loss (Interactive)")
                    t5_acc = gr.Plot(label="Grokking Accuracy (Interactive)")
                with gr.Row():
                    t5_phase_diff = gr.Image(label="Phase Difference |sin(D*)|", type="filepath")
                    t5_ipr = gr.Image(label="IPR & Parameter Norms", type="filepath")
                t5_memo = gr.Image(label="Memorization Accuracy", type="filepath")
                t5_memo_rare = gr.Image(label="Memorization: Common to Rare", type="filepath")
                t5_decoded = gr.Image(label="Decoded Weights Across Stages", type="filepath")

            # Tab 6: Lottery Mechanism
            with gr.Tab("Lottery Mechanism"):
                gr.Markdown(
                    "### Lottery Mechanism\n"
                    "How individual neurons specialize to specific frequencies. "
                    "Tracks magnitude and phase of a single neuron across all frequency modes."
                )
                t6_mag = gr.Image(label="Frequency Magnitude Evolution", type="filepath")
                t6_phase = gr.Image(label="Phase Misalignment Convergence", type="filepath")
                t6_contour = gr.Image(label="Final Magnitude Contour", type="filepath")

            # Tab 7: Gradient Dynamics
            with gr.Tab("Gradient Dynamics"):
                gr.Markdown(
                    "### Gradient Dynamics (Single-Frequency Initialization)\n"
                    "Phase alignment and magnitude evolution for Quad and ReLU activations."
                )
                with gr.Row():
                    t7_pa_quad = gr.Image(label="Phase Alignment (Quad)", type="filepath")
                    t7_sf_quad = gr.Image(label="Decoded Weights (Quad)", type="filepath")
                with gr.Row():
                    t7_pa_relu = gr.Image(label="Phase Alignment (ReLU)", type="filepath")
                    t7_sf_relu = gr.Image(label="Decoded Weights (ReLU)", type="filepath")

            # Tab 8: Theory
            with gr.Tab("Theory"):
                gr.Markdown(
                    "### Analytical & Simulation Results\n"
                    "Gradient flow simulations (no trained model needed) and "
                    "frequency/phase diversity analysis."
                )
                with gr.Row():
                    t8_approx1 = gr.Image(label="Gradient Flow Simulation (Case 1)", type="filepath")
                    t8_approx2 = gr.Image(label="Gradient Flow Simulation (Case 2)", type="filepath")
                with gr.Row():
                    t8_freq_div = gr.Image(label="Frequency Diversity", type="filepath")
                    t8_phase_div = gr.Image(label="Phase Diversity", type="filepath")

        # All outputs in order
        all_outputs = [
            info_md,
            # Tab 1
            t1_chart, t1_img,
            # Tab 2
            t2_heatmap, t2_line_in, t2_line_out,
            # Tab 3
            t3_phase_dist, t3_phase_rel, t3_magnitude,
            # Tab 4
            t4_logits,
            # Tab 5
            t5_loss, t5_acc, t5_phase_diff, t5_ipr,
            t5_memo, t5_memo_rare, t5_decoded,
            # Tab 6
            t6_mag, t6_phase, t6_contour,
            # Tab 7
            t7_pa_quad, t7_sf_quad, t7_pa_relu, t7_sf_relu,
            # Tab 8
            t8_approx1, t8_approx2, t8_freq_div, t8_phase_div,
        ]

        prime_dropdown.change(
            fn=on_prime_change,
            inputs=[prime_dropdown],
            outputs=all_outputs,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
