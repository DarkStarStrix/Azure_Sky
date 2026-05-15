"""
Dedicated large-scale Himmelblau benchmarking script.

Runs the generalised N-dimensional Himmelblau function across a wide
dimensionality sweep with full seed coverage, then generates a comprehensive
set of plots specifically designed to expose optimizer strengths and weaknesses
on this multi-modal, deceptive landscape.
"""
import sys
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.benchmarks.math_functions import HimmelblauND
from benchmarking.runners.suite_runner import _build_optimizer
from benchmarking.configs.suite_config import (
    OPTIMIZER_REGISTRY, HIMMELBLAU_DIM_SWEEP, INIT_RANGE,
)

RESULTS_DIR = Path(__file__).parent / 'results'
PLOTS_DIR   = Path(__file__).parent.parent / 'docs' / 'images' / 'benchmarking'
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_SEEDS = 10
STEPS     = 500

# Adaptive steps: high-dimensional runs use fewer steps to keep runtime manageable.
# dim <= 50  -> 500 steps (full resolution)
# dim <= 200 -> 300 steps
# dim > 200  -> 200 steps
DIM_STEPS = {2: 500, 4: 500, 8: 500, 16: 500, 32: 500, 50: 500,
             100: 300, 200: 200, 500: 150}

# ── Visual identity ───────────────────────────────────────────────────────────
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
    kwargs.setdefault('fontsize', 9)
    ax.legend(framealpha=0.15, facecolor=BG_PANEL, edgecolor=SPINE_COL,
              labelcolor=TEXT_COL, **kwargs)


def _save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ── Runner ────────────────────────────────────────────────────────────────────

def run_himmelblau_sweep():
    """Run the full Himmelblau dimensionality sweep and return structured results."""
    opt_names = list(OPTIMIZER_REGISTRY.keys())
    results = {opt: {} for opt in opt_names}
    total = len(opt_names) * len(HIMMELBLAU_DIM_SWEEP) * NUM_SEEDS
    done  = 0

    for opt in opt_names:
        for dim in HIMMELBLAU_DIM_SWEEP:
            results[opt][dim] = []
            for seed in range(NUM_SEEDS):
                torch.manual_seed(seed)
                np.random.seed(seed)

                bench = HimmelblauND(dim)
                x = torch.tensor(
                    np.random.uniform(INIT_RANGE[0], INIT_RANGE[1], size=dim),
                    dtype=torch.float32, requires_grad=True
                )
                optimizer = _build_optimizer(opt, [x])

                losses, dists = [], []
                t0 = time.perf_counter()

                n_steps = DIM_STEPS.get(dim, STEPS)
                for step in range(n_steps):
                    optimizer.zero_grad()
                    loss = bench.evaluate(x)
                    if torch.isnan(loss) or torch.isinf(loss):
                        losses.append(float('nan'))
                        dists.append(float('nan'))
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([x], max_norm=10.0)
                    optimizer.step()
                    losses.append(loss.item())
                    dists.append(bench.distance_to_global_min(x))

                elapsed = time.perf_counter() - t0
                final_loss = losses[-1] if losses else float('nan')
                final_dist = dists[-1]  if dists  else float('nan')

                results[opt][dim].append({
                    'seed':       seed,
                    'losses':     losses,
                    'dists':      dists,
                    'final_loss': final_loss,
                    'final_dist': final_dist,
                    'elapsed_s':  elapsed,
                    'converged':  final_loss < 1.0 if not np.isnan(final_loss) else False,
                })
                done += 1
                if done % 10 == 0:
                    print(f"  [{done}/{total}] {opt} | dim={dim:4d} | seed={seed} "
                          f"| loss={final_loss:10.4f} | dist={final_dist:.4f}", flush=True)

    # Serialise
    def _conv(o):
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, dict): return {str(k): _conv(v) for k, v in o.items()}
        if isinstance(o, list): return [_conv(i) for i in o]
        return o

    path = RESULTS_DIR / 'himmelblau_scale.json'
    with open(path, 'w') as f:
        json.dump(_conv(results), f, indent=2)
    print(f"\n  Results saved to {path}", flush=True)
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def _med_iqr(trials, key='losses'):
    arrays = [t[key] for t in trials
              if t[key] and not any(np.isnan(v) for v in t[key])]
    if not arrays:
        return None, None, None
    min_len = min(len(a) for a in arrays)
    mat = np.array([a[:min_len] for a in arrays])
    return (np.median(mat, 0), np.percentile(mat, 25, 0), np.percentile(mat, 75, 0))


def plot_convergence_grid(results):
    """
    Grid of convergence curves: one subplot per dimensionality, all optimizers.
    Shows the full 500-step trajectory with median ± IQR.
    """
    dims = HIMMELBLAU_DIM_SWEEP
    ncols = 3
    nrows = int(np.ceil(len(dims) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    fig.patch.set_facecolor(BG_DARK)
    axes = axes.flatten()

    for idx, dim in enumerate(dims):
        ax = axes[idx]
        _style_ax(ax)
        for opt, dim_map in results.items():
            trials = dim_map.get(dim, [])
            med, q25, q75 = _med_iqr(trials)
            if med is None:
                continue
            s = OPT_STYLE[opt]
            steps = np.arange(len(med))
            med_c = np.clip(med, 1e-6, None)
            ax.semilogy(steps, med_c, label=opt, color=s['color'],
                        lw=s['lw'], ls=s['ls'], zorder=s['zorder'])
            ax.fill_between(steps,
                            np.clip(q25, 1e-6, None),
                            np.clip(q75, 1e-6, None),
                            color=s['color'], alpha=0.12)
        ax.set_title(f'dim = {dim}', color=TEXT_COL, fontsize=10, fontweight='bold')
        ax.set_xlabel('Step', color=MUTED_COL, fontsize=8)
        ax.set_ylabel('Loss (log)', color=MUTED_COL, fontsize=8)
        if idx == 0:
            _legend(ax, fontsize=8)

    # Hide unused subplots
    for ax in axes[len(dims):]:
        ax.set_visible(False)

    plt.suptitle('Himmelblau N-D — Convergence Curves (median ± IQR, 10 seeds)',
                 color=TEXT_COL, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return _save(fig, 'himmelblau_convergence_grid.png')


def plot_final_loss_scaling(results):
    """
    Final loss vs dimensionality for all optimizers on a log-log scale.
    This is the primary scaling plot — shows how each optimizer degrades
    as the number of Himmelblau pairs increases.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG_DARK)
    _style_ax(ax)

    for opt, dim_map in results.items():
        dims_sorted = sorted(int(d) for d in dim_map.keys())
        medians, q25s, q75s = [], [], []
        for dim in dims_sorted:
            trials = dim_map.get(dim, [])
            finals = [t['final_loss'] for t in trials if not np.isnan(t['final_loss'])]
            medians.append(np.median(finals) if finals else float('nan'))
            q25s.append(np.percentile(finals, 25) if finals else float('nan'))
            q75s.append(np.percentile(finals, 75) if finals else float('nan'))

        s = OPT_STYLE[opt]
        d_arr = np.array(dims_sorted, dtype=float)
        m_arr = np.array(medians)
        valid = ~np.isnan(m_arr) & (m_arr > 0)
        ax.loglog(d_arr[valid], m_arr[valid],
                  label=opt, color=s['color'], lw=s['lw'],
                  ls=s['ls'], marker=s['marker'], markersize=7, zorder=s['zorder'])
        ax.fill_between(d_arr[valid],
                        np.clip(np.array(q25s)[valid], 1e-6, None),
                        np.clip(np.array(q75s)[valid], 1e-6, None),
                        color=s['color'], alpha=0.12)

    ax.set_title('Himmelblau N-D — Final Loss vs Dimensionality (log-log)',
                 color=TEXT_COL, fontsize=13, fontweight='bold')
    ax.set_xlabel('Dimensionality (log scale)', color=MUTED_COL, fontsize=11)
    ax.set_ylabel('Final Loss (log scale)', color=MUTED_COL, fontsize=11)
    _legend(ax)
    return _save(fig, 'himmelblau_final_loss_scaling.png')


def plot_distance_to_min_scaling(results):
    """
    Median distance to the nearest known global minimum vs dimensionality.
    Measures solution *quality* independent of absolute loss magnitude.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG_DARK)
    _style_ax(ax)

    for opt, dim_map in results.items():
        dims_sorted = sorted(int(d) for d in dim_map.keys())
        medians, q25s, q75s = [], [], []
        for dim in dims_sorted:
            trials = dim_map.get(dim, [])
            dists = [t['final_dist'] for t in trials if not np.isnan(t['final_dist'])]
            medians.append(np.median(dists) if dists else float('nan'))
            q25s.append(np.percentile(dists, 25) if dists else float('nan'))
            q75s.append(np.percentile(dists, 75) if dists else float('nan'))

        s = OPT_STYLE[opt]
        d_arr = np.array(dims_sorted, dtype=float)
        m_arr = np.array(medians)
        valid = ~np.isnan(m_arr)
        ax.semilogx(d_arr[valid], m_arr[valid],
                    label=opt, color=s['color'], lw=s['lw'],
                    ls=s['ls'], marker=s['marker'], markersize=7, zorder=s['zorder'])
        ax.fill_between(d_arr[valid],
                        np.array(q25s)[valid],
                        np.array(q75s)[valid],
                        color=s['color'], alpha=0.12)

    ax.set_title('Himmelblau N-D — Distance to Global Minimum vs Dimensionality',
                 color=TEXT_COL, fontsize=13, fontweight='bold')
    ax.set_xlabel('Dimensionality (log scale)', color=MUTED_COL, fontsize=11)
    ax.set_ylabel('Euclidean Distance to Nearest Global Min', color=MUTED_COL, fontsize=11)
    _legend(ax)
    return _save(fig, 'himmelblau_dist_to_min_scaling.png')


def plot_convergence_speed(results):
    """
    Step at which each optimizer first crosses loss thresholds [100, 10, 1, 0.1]
    at dim=2 and dim=50, side by side. Measures convergence speed vs quality.
    """
    thresholds = [100.0, 10.0, 1.0, 0.1]
    target_dims = [2, 50]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG_DARK)

    for ax, dim in zip(axes, target_dims):
        _style_ax(ax)
        opt_names = list(results.keys())
        x_pos = np.arange(len(thresholds))
        bar_w = 0.15

        for i, opt in enumerate(opt_names):
            trials = results[opt].get(dim, [])
            first_steps = []
            for thresh in thresholds:
                steps_list = []
                for t in trials:
                    step = next((s for s, l in enumerate(t['losses'])
                                 if not np.isnan(l) and l <= thresh), STEPS)
                    steps_list.append(step)
                first_steps.append(np.median(steps_list) if steps_list else STEPS)

            s = OPT_STYLE[opt]
            ax.bar(x_pos + i * bar_w, first_steps, bar_w,
                   label=opt, color=s['color'], alpha=0.85, zorder=3)

        ax.set_xticks(x_pos + bar_w * (len(opt_names) - 1) / 2)
        ax.set_xticklabels([f'loss ≤ {t}' for t in thresholds],
                           color=TEXT_COL, fontsize=9)
        ax.set_ylabel('Median Step to Threshold', color=MUTED_COL, fontsize=10)
        ax.set_title(f'Convergence Speed — dim={dim}',
                     color=TEXT_COL, fontsize=11, fontweight='bold')
        ax.axhline(STEPS, color=MUTED_COL, lw=1, ls='--', alpha=0.5)
        ax.text(len(thresholds) - 0.3, STEPS + 5, 'never', color=MUTED_COL, fontsize=8)
        _legend(ax, fontsize=8)

    plt.suptitle('Himmelblau — Steps to Reach Loss Threshold (lower = faster)',
                 color=TEXT_COL, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return _save(fig, 'himmelblau_convergence_speed.png')


def plot_reliability_heatmap(results):
    """
    Success rate heatmap: % of seeds converging to loss < 1.0 per (optimizer, dim).
    Exposes which optimizers are reliable vs lucky at each scale.
    """
    opt_names = list(results.keys())
    dims      = HIMMELBLAU_DIM_SWEEP

    matrix = np.zeros((len(opt_names), len(dims)))
    for oi, opt in enumerate(opt_names):
        for di, dim in enumerate(dims):
            trials = results[opt].get(dim, [])
            if trials:
                matrix[oi, di] = sum(t['converged'] for t in trials) / len(trials) * 100

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)

    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=MUTED_COL)
    cbar.set_label('Success Rate (%)', color=MUTED_COL, fontsize=10)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims], color=TEXT_COL, fontsize=10)
    ax.set_yticks(range(len(opt_names)))
    ax.set_yticklabels(opt_names, color=TEXT_COL, fontsize=11)
    ax.tick_params(colors=MUTED_COL)
    ax.set_xlabel('Dimensionality', color=MUTED_COL, fontsize=11)

    for oi in range(len(opt_names)):
        for di in range(len(dims)):
            val = matrix[oi, di]
            ax.text(di, oi, f'{val:.0f}%', ha='center', va='center',
                    color='white' if val < 55 else BG_DARK,
                    fontsize=9, fontweight='bold')

    ax.set_title('Himmelblau N-D — Convergence Reliability (loss < 1.0, 10 seeds)',
                 color=TEXT_COL, fontsize=13, fontweight='bold', pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)
    return _save(fig, 'himmelblau_reliability_heatmap.png')


def plot_variance_analysis(results):
    """
    IQR width (Q75 - Q25 of final loss) vs dimensionality.
    Measures optimizer *stability* — a wide IQR means the result is
    highly seed-dependent, which is a practical weakness in real workloads.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG_DARK)
    _style_ax(ax)

    for opt, dim_map in results.items():
        dims_sorted = sorted(int(d) for d in dim_map.keys())
        iqr_widths = []
        for dim in dims_sorted:
            trials = dim_map.get(dim, [])
            finals = [t['final_loss'] for t in trials if not np.isnan(t['final_loss'])]
            if len(finals) >= 4:
                iqr_widths.append(np.percentile(finals, 75) - np.percentile(finals, 25))
            else:
                iqr_widths.append(float('nan'))

        s = OPT_STYLE[opt]
        d_arr = np.array(dims_sorted, dtype=float)
        w_arr = np.array(iqr_widths)
        valid = ~np.isnan(w_arr)
        ax.semilogx(d_arr[valid], w_arr[valid],
                    label=opt, color=s['color'], lw=s['lw'],
                    ls=s['ls'], marker=s['marker'], markersize=7, zorder=s['zorder'])

    ax.set_title('Himmelblau N-D — Result Variance (IQR Width) vs Dimensionality',
                 color=TEXT_COL, fontsize=13, fontweight='bold')
    ax.set_xlabel('Dimensionality (log scale)', color=MUTED_COL, fontsize=11)
    ax.set_ylabel('IQR Width of Final Loss (lower = more stable)', color=MUTED_COL, fontsize=11)
    _legend(ax)
    return _save(fig, 'himmelblau_variance_analysis.png')


def plot_summary_panel(results):
    """
    Single-figure 2×3 summary panel combining the key metrics side by side
    for a complete at-a-glance view of all optimizer behaviour on Himmelblau.
    """
    dims_sorted = sorted(int(d) for d in next(iter(results.values())).keys())
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(BG_DARK)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: Final loss scaling (log-log) ─────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _style_ax(ax_a)
    for opt, dim_map in results.items():
        medians = []
        for dim in dims_sorted:
            finals = [t['final_loss'] for t in dim_map.get(dim, [])
                      if not np.isnan(t['final_loss'])]
            medians.append(np.median(finals) if finals else float('nan'))
        s = OPT_STYLE[opt]
        d = np.array(dims_sorted, dtype=float)
        m = np.array(medians)
        v = ~np.isnan(m) & (m > 0)
        ax_a.loglog(d[v], m[v], label=opt, color=s['color'], lw=s['lw'],
                    ls=s['ls'], marker=s['marker'], markersize=5, zorder=s['zorder'])
    ax_a.set_title('A — Final Loss Scaling', color=TEXT_COL, fontsize=10, fontweight='bold')
    ax_a.set_xlabel('Dim', color=MUTED_COL, fontsize=9)
    ax_a.set_ylabel('Loss (log)', color=MUTED_COL, fontsize=9)
    _legend(ax_a, fontsize=7)

    # ── Panel B: Distance to global min ──────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _style_ax(ax_b)
    for opt, dim_map in results.items():
        medians = []
        for dim in dims_sorted:
            dists = [t['final_dist'] for t in dim_map.get(dim, [])
                     if not np.isnan(t['final_dist'])]
            medians.append(np.median(dists) if dists else float('nan'))
        s = OPT_STYLE[opt]
        d = np.array(dims_sorted, dtype=float)
        m = np.array(medians)
        v = ~np.isnan(m)
        ax_b.semilogx(d[v], m[v], label=opt, color=s['color'], lw=s['lw'],
                      ls=s['ls'], marker=s['marker'], markersize=5, zorder=s['zorder'])
    ax_b.set_title('B — Distance to Global Min', color=TEXT_COL, fontsize=10, fontweight='bold')
    ax_b.set_xlabel('Dim', color=MUTED_COL, fontsize=9)
    ax_b.set_ylabel('Euclidean Distance', color=MUTED_COL, fontsize=9)
    _legend(ax_b, fontsize=7)

    # ── Panel C: Variance (IQR width) ─────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    _style_ax(ax_c)
    for opt, dim_map in results.items():
        iqrs = []
        for dim in dims_sorted:
            finals = [t['final_loss'] for t in dim_map.get(dim, [])
                      if not np.isnan(t['final_loss'])]
            iqrs.append((np.percentile(finals, 75) - np.percentile(finals, 25))
                        if len(finals) >= 4 else float('nan'))
        s = OPT_STYLE[opt]
        d = np.array(dims_sorted, dtype=float)
        w = np.array(iqrs)
        v = ~np.isnan(w)
        ax_c.semilogx(d[v], w[v], label=opt, color=s['color'], lw=s['lw'],
                      ls=s['ls'], marker=s['marker'], markersize=5, zorder=s['zorder'])
    ax_c.set_title('C — Result Variance (IQR)', color=TEXT_COL, fontsize=10, fontweight='bold')
    ax_c.set_xlabel('Dim', color=MUTED_COL, fontsize=9)
    ax_c.set_ylabel('IQR Width', color=MUTED_COL, fontsize=9)
    _legend(ax_c, fontsize=7)

    # ── Panel D: Convergence at dim=2 ─────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    _style_ax(ax_d)
    for opt, dim_map in results.items():
        trials = dim_map.get(2, [])
        med, q25, q75 = _med_iqr(trials)
        if med is None: continue
        s = OPT_STYLE[opt]
        steps = np.arange(len(med))
        ax_d.semilogy(steps, np.clip(med, 1e-6, None), label=opt,
                      color=s['color'], lw=s['lw'], ls=s['ls'], zorder=s['zorder'])
        ax_d.fill_between(steps, np.clip(q25, 1e-6, None), np.clip(q75, 1e-6, None),
                          color=s['color'], alpha=0.12)
    ax_d.set_title('D — Convergence at dim=2', color=TEXT_COL, fontsize=10, fontweight='bold')
    ax_d.set_xlabel('Step', color=MUTED_COL, fontsize=9)
    ax_d.set_ylabel('Loss (log)', color=MUTED_COL, fontsize=9)
    _legend(ax_d, fontsize=7)

    # ── Panel E: Convergence at dim=100 ──────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    _style_ax(ax_e)
    for opt, dim_map in results.items():
        trials = dim_map.get(100, [])
        med, q25, q75 = _med_iqr(trials)
        if med is None: continue
        s = OPT_STYLE[opt]
        steps = np.arange(len(med))
        ax_e.semilogy(steps, np.clip(med, 1e-6, None), label=opt,
                      color=s['color'], lw=s['lw'], ls=s['ls'], zorder=s['zorder'])
        ax_e.fill_between(steps, np.clip(q25, 1e-6, None), np.clip(q75, 1e-6, None),
                          color=s['color'], alpha=0.12)
    ax_e.set_title('E — Convergence at dim=100', color=TEXT_COL, fontsize=10, fontweight='bold')
    ax_e.set_xlabel('Step', color=MUTED_COL, fontsize=9)
    ax_e.set_ylabel('Loss (log)', color=MUTED_COL, fontsize=9)
    _legend(ax_e, fontsize=7)

    # ── Panel F: Reliability heatmap (compact) ────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.set_facecolor(BG_DARK)
    opt_names = list(results.keys())
    matrix = np.zeros((len(opt_names), len(dims_sorted)))
    for oi, opt in enumerate(opt_names):
        for di, dim in enumerate(dims_sorted):
            trials = results[opt].get(dim, [])
            if trials:
                matrix[oi, di] = sum(t['converged'] for t in trials) / len(trials) * 100
    im = ax_f.imshow(matrix, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    ax_f.set_xticks(range(len(dims_sorted)))
    ax_f.set_xticklabels([str(d) for d in dims_sorted], color=TEXT_COL, fontsize=7, rotation=45)
    ax_f.set_yticks(range(len(opt_names)))
    ax_f.set_yticklabels(opt_names, color=TEXT_COL, fontsize=9)
    ax_f.tick_params(colors=MUTED_COL)
    for oi in range(len(opt_names)):
        for di in range(len(dims_sorted)):
            val = matrix[oi, di]
            ax_f.text(di, oi, f'{val:.0f}', ha='center', va='center',
                      color='white' if val < 55 else BG_DARK, fontsize=7, fontweight='bold')
    ax_f.set_title('F — Reliability (%)', color=TEXT_COL, fontsize=10, fontweight='bold')
    for spine in ax_f.spines.values():
        spine.set_edgecolor(SPINE_COL)

    plt.suptitle('Himmelblau N-D — Complete Performance Summary  |  AzureSky vs Adam vs AdamW vs SGD vs RMSprop',
                 color=TEXT_COL, fontsize=13, fontweight='bold', y=1.01)
    return _save(fig, 'himmelblau_summary_panel.png')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("Himmelblau N-D Large-Scale Benchmark")
    print(f"Dims    : {HIMMELBLAU_DIM_SWEEP}")
    print(f"Seeds   : {NUM_SEEDS}")
    print(f"Steps   : {STEPS}")
    print(f"Optimizers: {list(OPTIMIZER_REGISTRY.keys())}")
    print("=" * 65)

    results = run_himmelblau_sweep()

    print("\nGenerating plots...", flush=True)
    plot_convergence_grid(results)
    plot_final_loss_scaling(results)
    plot_distance_to_min_scaling(results)
    plot_convergence_speed(results)
    plot_reliability_heatmap(results)
    plot_variance_analysis(results)
    plot_summary_panel(results)

    print("\nAll done. Plots saved to docs/images/benchmarking/", flush=True)
