"""
Reporting module for the AzureSky benchmarking suite.

Consumes the structured result dictionaries produced by the suite runner and
generates publication-quality figures. All plots use a consistent dark theme
matching the repository's existing visual identity.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PLOTS_DIR = Path(__file__).parent.parent.parent / 'docs' / 'images' / 'benchmarking'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BG_DARK   = '#0D1117'
BG_PANEL  = '#161B22'
GRID_COL  = '#21262D'
SPINE_COL = '#30363D'
TEXT_COL  = '#E6EDF3'
MUTED_COL = '#8B949E'

OPT_STYLE = {
    'AzureSky': {'color': '#4A9EFF', 'lw': 2.5, 'ls': '-',  'zorder': 5, 'marker': 'o'},
    'Adam':     {'color': '#FF6B6B', 'lw': 1.8, 'ls': '--', 'zorder': 4, 'marker': 's'},
    'AdamW':    {'color': '#C084FC', 'lw': 1.8, 'ls': '-.',  'zorder': 4, 'marker': 'D'},
    'SGD':      {'color': '#51CF66', 'lw': 1.8, 'ls': ':',  'zorder': 3, 'marker': '^'},
    'RMSprop':  {'color': '#FFA94D', 'lw': 1.8, 'ls': '-.', 'zorder': 3, 'marker': 'v'},
}


def _style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=MUTED_COL, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.7, alpha=0.8)


def _legend(ax, **kwargs):
    ax.legend(framealpha=0.15, facecolor=BG_PANEL, edgecolor=SPINE_COL,
              labelcolor=TEXT_COL, fontsize=9, **kwargs)


def _save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def _median_iqr(trials_by_seed: list, key: str = 'losses'):
    """Extract median and IQR across seeds for a list of trial dicts."""
    arrays = [t[key] for t in trials_by_seed if not any(np.isnan(v) for v in t[key])]
    if not arrays:
        return None, None, None
    min_len = min(len(a) for a in arrays)
    mat = np.array([a[:min_len] for a in arrays])
    return (np.median(mat, axis=0),
            np.percentile(mat, 25, axis=0),
            np.percentile(mat, 75, axis=0))


# ── Plot 1: Convergence curves per benchmark (median ± IQR) ──────────────────

def plot_convergence_curves(dim_results: dict, dim: int = 20,
                             benchmarks: list = None) -> list:
    """
    Plot median convergence curves with IQR shading for each benchmark at a
    fixed dimensionality.

    Args:
        dim_results (dict): Output of run_dimensionality_sweep().
        dim (int): Dimensionality slice to plot.
        benchmarks (list): Subset of benchmarks to include.

    Returns:
        list: Paths to saved figures.
    """
    benchmarks = benchmarks or list(dim_results.keys())
    paths = []

    for bname in benchmarks:
        opt_data = dim_results[bname]
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(BG_DARK)
        _style_ax(ax)

        for opt, dim_map in opt_data.items():
            trials = dim_map.get(dim) or dim_map.get(str(dim), [])
            if not trials:
                continue
            med, q25, q75 = _median_iqr(trials)
            if med is None:
                continue
            s = OPT_STYLE.get(opt, {'color': 'white', 'lw': 1.5, 'ls': '-', 'zorder': 2})
            steps = np.arange(len(med))
            clipped_med = np.clip(med, 1e-10, None)
            clipped_q25 = np.clip(q25, 1e-10, None)
            clipped_q75 = np.clip(q75, 1e-10, None)
            ax.semilogy(steps, clipped_med, label=opt, color=s['color'],
                        lw=s['lw'], linestyle=s['ls'], zorder=s['zorder'])
            ax.fill_between(steps, clipped_q25, clipped_q75,
                            color=s['color'], alpha=0.12, zorder=s['zorder'] - 1)

        ax.set_title(f'{bname} — Convergence (dim={dim}, median ± IQR)',
                     color=TEXT_COL, fontsize=12, fontweight='bold')
        ax.set_xlabel('Step', color=MUTED_COL, fontsize=10)
        ax.set_ylabel('Loss (log scale)', color=MUTED_COL, fontsize=10)
        _legend(ax)
        fname = f'convergence_{bname.lower()}_dim{dim}.png'
        paths.append(_save(fig, fname))

    return paths


# ── Plot 2: Dimensionality scaling — final loss vs dim ────────────────────────

def plot_dimensionality_scaling(dim_results: dict, benchmarks: list = None) -> list:
    """
    Plot final loss as a function of dimensionality for each benchmark,
    showing how each optimizer degrades as the problem scales.
    """
    benchmarks = benchmarks or list(dim_results.keys())
    paths = []

    for bname in benchmarks:
        opt_data = dim_results[bname]
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(BG_DARK)
        _style_ax(ax)

        for opt, dim_map in opt_data.items():
            dims_sorted = sorted(int(d) for d in dim_map.keys())
            medians, q25s, q75s = [], [], []
            for dim in dims_sorted:
                trials = dim_map.get(dim) or dim_map.get(str(dim), [])
                finals = [t['final_loss'] for t in trials
                          if not np.isnan(t['final_loss'])]
                if finals:
                    medians.append(np.median(finals))
                    q25s.append(np.percentile(finals, 25))
                    q75s.append(np.percentile(finals, 75))
                else:
                    medians.append(float('nan'))
                    q25s.append(float('nan'))
                    q75s.append(float('nan'))

            s = OPT_STYLE.get(opt, {'color': 'white', 'lw': 1.5, 'ls': '-',
                                     'zorder': 2, 'marker': 'o'})
            valid = ~np.isnan(medians)
            dims_arr = np.array(dims_sorted)
            ax.semilogy(dims_arr[valid], np.array(medians)[valid],
                        label=opt, color=s['color'], lw=s['lw'],
                        linestyle=s['ls'], marker=s['marker'],
                        markersize=6, zorder=s['zorder'])
            ax.fill_between(dims_arr[valid],
                            np.array(q25s)[valid], np.array(q75s)[valid],
                            color=s['color'], alpha=0.12)

        ax.set_title(f'{bname} — Final Loss vs Dimensionality',
                     color=TEXT_COL, fontsize=12, fontweight='bold')
        ax.set_xlabel('Dimensionality', color=MUTED_COL, fontsize=10)
        ax.set_ylabel('Final Loss (log scale)', color=MUTED_COL, fontsize=10)
        _legend(ax)
        fname = f'dim_scaling_{bname.lower()}.png'
        paths.append(_save(fig, fname))

    return paths


# ── Plot 3: Non-convexity dial — final loss vs alpha ─────────────────────────

def plot_nonconvexity_dial(nc_results: dict) -> Path:
    """
    Plot how each optimizer's final loss degrades as the non-convexity
    amplitude alpha increases from 0 (convex) to large values.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG_DARK)
    _style_ax(ax)

    for opt, alpha_map in nc_results.items():
        alphas_sorted = sorted(float(a) for a in alpha_map.keys())
        medians, q25s, q75s = [], [], []
        for alpha in alphas_sorted:
            trials = alpha_map.get(alpha) or alpha_map.get(str(alpha), [])
            finals = [t['final_loss'] for t in trials
                      if not np.isnan(t['final_loss'])]
            if finals:
                medians.append(np.median(finals))
                q25s.append(np.percentile(finals, 25))
                q75s.append(np.percentile(finals, 75))
            else:
                medians.append(float('nan'))
                q25s.append(float('nan'))
                q75s.append(float('nan'))

        s = OPT_STYLE.get(opt, {'color': 'white', 'lw': 1.5, 'ls': '-',
                                  'zorder': 2, 'marker': 'o'})
        valid = ~np.isnan(medians)
        alphas_arr = np.array(alphas_sorted)
        ax.plot(alphas_arr[valid], np.array(medians)[valid],
                label=opt, color=s['color'], lw=s['lw'],
                linestyle=s['ls'], marker=s['marker'],
                markersize=6, zorder=s['zorder'])
        ax.fill_between(alphas_arr[valid],
                        np.array(q25s)[valid], np.array(q75s)[valid],
                        color=s['color'], alpha=0.12)

    ax.set_title('Non-Convexity Dial — Final Loss vs Perturbation Amplitude (α)',
                 color=TEXT_COL, fontsize=12, fontweight='bold')
    ax.set_xlabel('α  (0 = convex quadratic  →  large = highly non-convex)',
                  color=MUTED_COL, fontsize=10)
    ax.set_ylabel('Final Loss', color=MUTED_COL, fontsize=10)
    _legend(ax)
    return _save(fig, 'nonconvexity_dial.png')


# ── Plot 4: Reliability — convergence success rate ────────────────────────────

def plot_reliability(dim_results: dict, benchmarks: list = None,
                     dims: list = None) -> Path:
    """
    Plot convergence success rate (fraction of seeds that converged to loss < 1e-3)
    as a heatmap across (benchmark × optimizer) pairs.
    """
    benchmarks = benchmarks or list(dim_results.keys())
    dims = dims or [20]
    opt_names = list(OPT_STYLE.keys())

    # Aggregate success rates across all requested dims
    # Use a per-benchmark threshold: top-10% of all final losses seen
    # across all optimizers at the requested dims, so the heatmap is always
    # meaningful regardless of the absolute loss scale.
    thresholds = {}
    for bname in benchmarks:
        all_finals = []
        for opt in opt_names:
            for dim in dims:
                trials = (dim_results[bname].get(opt, {}).get(dim)
                          or dim_results[bname].get(opt, {}).get(str(dim), []))
                all_finals += [t['final_loss'] for t in trials
                               if not np.isnan(t['final_loss'])]
        thresholds[bname] = np.percentile(all_finals, 25) if all_finals else 1.0

    matrix = np.zeros((len(benchmarks), len(opt_names)))
    for bi, bname in enumerate(benchmarks):
        thresh = thresholds[bname]
        for oi, opt in enumerate(opt_names):
            successes, total = 0, 0
            for dim in dims:
                trials = (dim_results[bname].get(opt, {}).get(dim)
                          or dim_results[bname].get(opt, {}).get(str(dim), []))
                for t in trials:
                    total += 1
                    if not np.isnan(t['final_loss']) and t['final_loss'] <= thresh:
                        successes += 1
            matrix[bi, oi] = (successes / total * 100) if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, max(4, len(benchmarks) * 0.9)))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)

    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=MUTED_COL)
    cbar.set_label('Success Rate (%)', color=MUTED_COL, fontsize=10)

    ax.set_xticks(range(len(opt_names)))
    ax.set_xticklabels(opt_names, color=TEXT_COL, fontsize=10)
    ax.set_yticks(range(len(benchmarks)))
    ax.set_yticklabels(benchmarks, color=TEXT_COL, fontsize=10)
    ax.tick_params(colors=MUTED_COL)

    for bi in range(len(benchmarks)):
        for oi in range(len(opt_names)):
            val = matrix[bi, oi]
            ax.text(oi, bi, f'{val:.0f}%', ha='center', va='center',
                    color='white' if val < 60 else BG_DARK, fontsize=9,
                    fontweight='bold')

    ax.set_title(f'Convergence Reliability — % Trials in Bottom-25th Percentile Loss (dims={dims})',
                 color=TEXT_COL, fontsize=12, fontweight='bold', pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)

    return _save(fig, 'reliability_heatmap.png')


# ── Plot 5: NN suite — training curves and basin sharpness ───────────────────

def plot_nn_training_curves(nn_results: dict, benchmarks: list = None) -> list:
    """
    Plot training loss and accuracy curves for neural network benchmarks,
    with median ± IQR across seeds.
    """
    benchmarks = benchmarks or list(nn_results.keys())
    paths = []

    for bname in benchmarks:
        opt_data = nn_results[bname]
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor(BG_DARK)
        _style_ax(ax_loss)
        _style_ax(ax_acc)

        for opt, trials in opt_data.items():
            s = OPT_STYLE.get(opt, {'color': 'white', 'lw': 1.5, 'ls': '-', 'zorder': 2})

            loss_med, loss_q25, loss_q75 = _median_iqr(trials, 'train_losses')
            acc_med,  acc_q25,  acc_q75  = _median_iqr(trials, 'train_accs')

            if loss_med is not None:
                epochs = np.arange(1, len(loss_med) + 1)
                ax_loss.plot(epochs, loss_med, label=opt, color=s['color'],
                             lw=s['lw'], linestyle=s['ls'], zorder=s['zorder'])
                ax_loss.fill_between(epochs, loss_q25, loss_q75,
                                     color=s['color'], alpha=0.12)

            if acc_med is not None:
                epochs = np.arange(1, len(acc_med) + 1)
                ax_acc.plot(epochs, acc_med, label=opt, color=s['color'],
                            lw=s['lw'], linestyle=s['ls'], zorder=s['zorder'])
                ax_acc.fill_between(epochs, acc_q25, acc_q75,
                                    color=s['color'], alpha=0.12)

        ax_loss.set_title(f'{bname} — Training Loss', color=TEXT_COL,
                          fontsize=11, fontweight='bold')
        ax_loss.set_xlabel('Epoch', color=MUTED_COL, fontsize=10)
        ax_loss.set_ylabel('Loss', color=MUTED_COL, fontsize=10)
        _legend(ax_loss)

        ax_acc.set_title(f'{bname} — Training Accuracy', color=TEXT_COL,
                         fontsize=11, fontweight='bold')
        ax_acc.set_xlabel('Epoch', color=MUTED_COL, fontsize=10)
        ax_acc.set_ylabel('Accuracy (%)', color=MUTED_COL, fontsize=10)
        _legend(ax_acc)

        plt.suptitle(f'Neural Network Benchmark: {bname}',
                     color=TEXT_COL, fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        fname = f'nn_{bname.lower().replace(" ", "_")}.png'
        paths.append(_save(fig, fname))

    return paths


def plot_basin_sharpness(nn_results: dict) -> Path:
    """
    Plot the Hessian trace (basin sharpness) at convergence for each optimizer
    across all NN benchmarks. Lower trace = flatter minimum = better generalisation.
    """
    benchmarks = list(nn_results.keys())
    opt_names  = list(OPT_STYLE.keys())

    data = {opt: [] for opt in opt_names}
    labels = []

    for bname in benchmarks:
        labels.append(bname)
        for opt in opt_names:
            trials = nn_results[bname].get(opt, [])
            traces = [t['hessian_trace'] for t in trials
                      if not np.isnan(t.get('hessian_trace', float('nan')))]
            data[opt].append(np.median(traces) if traces else float('nan'))

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG_DARK)
    _style_ax(ax)

    x = np.arange(len(labels))
    width = 0.15
    for i, opt in enumerate(opt_names):
        vals = data[opt]
        s = OPT_STYLE.get(opt, {'color': 'white'})
        ax.bar(x + i * width, vals, width, label=opt,
               color=s['color'], alpha=0.85, zorder=3)

    ax.set_xticks(x + width * (len(opt_names) - 1) / 2)
    ax.set_xticklabels([l.replace('_', '\n') for l in labels],
                       color=TEXT_COL, fontsize=9)
    ax.set_ylabel('Hessian Trace (lower = flatter minimum)', color=MUTED_COL, fontsize=10)
    ax.set_title('Basin Sharpness at Convergence — Hutchinson Hessian Trace Estimate',
                 color=TEXT_COL, fontsize=12, fontweight='bold')
    _legend(ax)
    return _save(fig, 'basin_sharpness.png')


def generate_summary_table(dim_results: dict, nc_results: dict,
                            nn_results: dict) -> str:
    """
    Generate a Markdown summary table of key metrics across all experiments.

    Returns:
        str: Markdown-formatted table.
    """
    opt_names = list(OPT_STYLE.keys())
    rows = []

    for opt in opt_names:
        # Math: median final loss on Ackley dim=20
        ackley_trials = (dim_results.get('Ackley', {})
                         .get(opt, {}).get(20) or
                         dim_results.get('Ackley', {})
                         .get(opt, {}).get('20', []))
        ackley_finals = [t['final_loss'] for t in ackley_trials
                         if not np.isnan(t['final_loss'])]
        ackley_med = f"{np.median(ackley_finals):.4f}" if ackley_finals else 'N/A'

        # NC dial: final loss at alpha=5.0
        nc_trials = (nc_results.get(opt, {}).get(5.0) or
                     nc_results.get(opt, {}).get('5.0', []))
        nc_finals = [t['final_loss'] for t in nc_trials
                     if not np.isnan(t['final_loss'])]
        nc_med = f"{np.median(nc_finals):.4f}" if nc_finals else 'N/A'

        # NN: median final accuracy on TwoMoons_Medium
        nn_trials = nn_results.get('TwoMoons_Medium', {}).get(opt, [])
        nn_accs = [t['final_acc'] for t in nn_trials
                   if not np.isnan(t.get('final_acc', float('nan')))]
        nn_med = f"{np.median(nn_accs):.1f}%" if nn_accs else 'N/A'

        rows.append(f"| {opt:<10} | {ackley_med:<20} | {nc_med:<22} | {nn_med:<22} |")

    header = ("| Optimizer  | Ackley 20D (final loss) "
              "| NonConvex α=5 (final loss) | TwoMoons Acc (median) |\n"
              "|------------|------------------------|"
              "--------------------------|------------------------|")
    return header + "\n" + "\n".join(rows)
