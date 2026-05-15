"""
Generate performance plots for the README.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from Backend.Benchmarks.Himmelblau import Himmelblau
from Backend.Benchmarks.Ackley import AckleyN2
from Backend.Benchmarks.Adjiman import Adjiman
from Backend.optimizers.azure_optim import Azure
from Backend.optimizers.adam import AdamOptimizer
from Backend.optimizers.SGD import SGDOptimizer
from Backend.optimizers.RMSprop import RMSpropOptimizer

os.makedirs("docs/images", exist_ok=True)

STEPS = 500
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

STYLE = {
    'AzureSky': {'color': '#4A9EFF', 'lw': 2.5, 'ls': '-',  'zorder': 5},
    'Adam':     {'color': '#FF6B6B', 'lw': 1.8, 'ls': '--', 'zorder': 4},
    'SGD':      {'color': '#51CF66', 'lw': 1.8, 'ls': ':',  'zorder': 3},
    'RMSprop':  {'color': '#FFA94D', 'lw': 1.8, 'ls': '-.', 'zorder': 3},
}

def run_benchmark(benchmark_class, optimizer_configs, steps=STEPS, init=None):
    """Run all optimizers on a benchmark and return loss histories."""
    results = {}
    for name, (opt_class, kwargs) in optimizer_configs.items():
        benchmark = benchmark_class()
        dim = benchmark.dimensions
        if init is not None:
            x0 = torch.tensor(init, dtype=torch.float32)
        else:
            torch.manual_seed(SEED)
            x0 = torch.tensor(np.random.uniform(-3, 3, size=dim), dtype=torch.float32)
        x = x0.clone().requires_grad_(True)
        opt = opt_class([x], **kwargs)
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            loss = benchmark.evaluate(x)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        results[name] = losses
    return results

OPTIMIZERS = {
    'AzureSky': (Azure,          {'lr': 0.05, 'sa_steps': 200}),
    'Adam':     (AdamOptimizer,  {'lr': 0.05}),
    'SGD':      (SGDOptimizer,   {'lr': 0.02, 'momentum': 0.9}),
    'RMSprop':  (RMSpropOptimizer, {'lr': 0.02}),
}

print("Running Himmelblau benchmark...")
himmelblau_results = run_benchmark(Himmelblau, OPTIMIZERS, init=[1.0, 1.0])

print("Running Ackley benchmark...")
ackley_opts = {
    'AzureSky': (Azure,          {'lr': 0.02, 'sa_steps': 200}),
    'Adam':     (AdamOptimizer,  {'lr': 0.02}),
    'SGD':      (SGDOptimizer,   {'lr': 0.005, 'momentum': 0.9}),
    'RMSprop':  (RMSpropOptimizer, {'lr': 0.005}),
}
ackley_results = run_benchmark(AckleyN2, ackley_opts, init=[2.0]*10)

print("Running Adjiman benchmark...")
adjiman_results = run_benchmark(Adjiman, OPTIMIZERS, init=[2.0, 2.0])

# ── Plot 1: Convergence curves (3 benchmarks) ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0D1117')

benchmark_data = [
    ('Himmelblau', himmelblau_results),
    ('Ackley N2 (10D)', ackley_results),
    ('Adjiman', adjiman_results),
]

for ax, (title, results) in zip(axes, benchmark_data):
    ax.set_facecolor('#161B22')
    for name, losses in results.items():
        s = STYLE[name]
        clipped = np.clip(losses, 1e-8, None)
        ax.semilogy(clipped, label=name, color=s['color'], lw=s['lw'],
                    linestyle=s['ls'], zorder=s['zorder'])
    ax.set_title(title, color='#E6EDF3', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Step', color='#8B949E', fontsize=10)
    ax.set_ylabel('Loss (log scale)', color='#8B949E', fontsize=10)
    ax.tick_params(colors='#8B949E', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')
    ax.grid(True, color='#21262D', linewidth=0.7, alpha=0.8)
    ax.legend(framealpha=0.15, facecolor='#161B22', edgecolor='#30363D',
              labelcolor='#E6EDF3', fontsize=9)

plt.suptitle('AzureSky Optimizer — Benchmark Convergence', color='#E6EDF3',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/images/benchmark_convergence.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("Saved benchmark_convergence.png")

# ── Plot 2: Final loss bar chart ───────────────────────────────────────────
benchmarks_final = {
    'Himmelblau': {k: v[-1] for k, v in himmelblau_results.items()},
    'Ackley N2':  {k: v[-1] for k, v in ackley_results.items()},
    'Adjiman':    {k: v[-1] for k, v in adjiman_results.items()},
}

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#161B22')

opt_names = list(OPTIMIZERS.keys())
bench_names = list(benchmarks_final.keys())
x_pos = np.arange(len(bench_names))
width = 0.2

for i, opt in enumerate(opt_names):
    vals = [max(benchmarks_final[b].get(opt, 1e-8), 1e-8) for b in bench_names]
    bars = ax.bar(x_pos + i * width, vals, width, label=opt,
                  color=STYLE[opt]['color'], alpha=0.85, zorder=3)

ax.set_yscale('log')
ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels(bench_names, color='#E6EDF3', fontsize=11)
ax.set_ylabel('Final Loss (log scale)', color='#8B949E', fontsize=11)
ax.set_title('Final Loss After 500 Steps — All Optimizers', color='#E6EDF3',
             fontsize=13, fontweight='bold')
ax.tick_params(colors='#8B949E', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#30363D')
ax.grid(True, axis='y', color='#21262D', linewidth=0.7, alpha=0.8)
ax.legend(framealpha=0.15, facecolor='#161B22', edgecolor='#30363D',
          labelcolor='#E6EDF3', fontsize=10)

plt.tight_layout()
plt.savefig('docs/images/final_loss_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("Saved final_loss_comparison.png")

# ── Plot 3: Himmelblau 2D path visualization ───────────────────────────────
def run_with_path(benchmark_class, opt_class, kwargs, init, steps=300):
    benchmark = benchmark_class()
    x = torch.tensor(init, dtype=torch.float32, requires_grad=True)
    opt = opt_class([x], **kwargs)
    path = [x.detach().numpy().copy()]
    for _ in range(steps):
        opt.zero_grad()
        loss = benchmark.evaluate(x)
        loss.backward()
        opt.step()
        path.append(x.detach().numpy().copy())
    return np.array(path)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('#0D1117')

path_configs = [
    ('AzureSky', Azure, {'lr': 0.05, 'sa_steps': 150}),
    ('Adam',     AdamOptimizer, {'lr': 0.05}),
]

# Himmelblau surface
x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2

for ax, (name, opt_class, kwargs) in zip(axes, path_configs):
    ax.set_facecolor('#0D1117')
    contour = ax.contourf(X, Y, np.log1p(Z), levels=40, cmap='Blues_r', alpha=0.85)
    ax.contour(X, Y, Z, levels=[0.1, 1, 5, 20, 60], colors='#30363D', linewidths=0.5, alpha=0.6)
    
    path = run_with_path(Himmelblau, opt_class, kwargs, [1.0, 1.0])
    ax.plot(path[:, 0], path[:, 1], color=STYLE[name]['color'], lw=1.8,
            alpha=0.9, zorder=4)
    ax.scatter(path[0, 0], path[0, 1], color='white', s=60, zorder=6, label='Start')
    ax.scatter(path[-1, 0], path[-1, 1], color=STYLE[name]['color'], s=80,
               marker='*', zorder=6, label='End')
    
    # Mark known minima
    for mx, my in [(3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]:
        ax.scatter(mx, my, color='#FFD700', s=50, marker='x', zorder=5, linewidths=1.5)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(f'{name} — Himmelblau Path', color='#E6EDF3', fontsize=12, fontweight='bold')
    ax.set_xlabel('x₁', color='#8B949E', fontsize=10)
    ax.set_ylabel('x₂', color='#8B949E', fontsize=10)
    ax.tick_params(colors='#8B949E', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363D')
    ax.legend(framealpha=0.2, facecolor='#161B22', edgecolor='#30363D',
              labelcolor='#E6EDF3', fontsize=9, loc='upper right')

plt.suptitle('Optimization Paths on Himmelblau Function  (★ = known minima)',
             color='#E6EDF3', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('docs/images/himmelblau_paths.png', dpi=150, bbox_inches='tight',
            facecolor='#0D1117')
plt.close()
print("Saved himmelblau_paths.png")

# ── Print final loss table ─────────────────────────────────────────────────
print("\n=== Final Loss Summary (after 500 steps) ===")
print(f"{'Optimizer':<12}", end="")
for b in bench_names:
    print(f"  {b:<14}", end="")
print()
for opt in opt_names:
    print(f"{opt:<12}", end="")
    for b in bench_names:
        val = benchmarks_final[b].get(opt, float('nan'))
        print(f"  {val:<14.6f}", end="")
    print()

print("\nAll plots saved to docs/images/")
