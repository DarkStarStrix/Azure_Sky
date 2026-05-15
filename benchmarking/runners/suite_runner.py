"""
Unified suite runner for the AzureSky benchmarking suite.

Handles both mathematical benchmark trials and neural network landscape trials.
All optimizers are instantiated from the central config with identical shared
hyperparameters. Results are collected into structured dictionaries and
serialised to JSON for downstream reporting.
"""
import importlib
import json
import time
import numpy as np
import torch
from pathlib import Path

from benchmarking.configs.suite_config import (
    OPTIMIZER_REGISTRY, NUM_SEEDS, STEPS, INIT_RANGE,
    DIMENSIONALITY_SWEEP, NONCONVEXITY_SWEEP, MATH_BENCHMARK_REGISTRY,
    NN_BENCHMARK_REGISTRY,
)
from benchmarking.benchmarks.math_functions import BENCHMARK_MAP, NonConvexDial
from benchmarking.benchmarks.nn_landscapes import NNLandmarkBenchmark

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def _build_optimizer(opt_name: str, params):
    """
    Instantiate an optimizer from the registry with its configured kwargs.

    Args:
        opt_name (str): Key in OPTIMIZER_REGISTRY.
        params: Iterable of parameters to optimise.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.
    """
    cfg = OPTIMIZER_REGISTRY[opt_name]
    mod = importlib.import_module(cfg['module'])
    cls = getattr(mod, cfg['class'])
    return cls(params, **cfg['kwargs'])


def run_math_trial(benchmark_cls, dim: int, opt_name: str, seed: int,
                   steps: int = STEPS) -> dict:
    """
    Run a single mathematical benchmark trial.

    Args:
        benchmark_cls: Benchmark class (e.g. AckleyND).
        dim (int): Dimensionality.
        opt_name (str): Optimizer name.
        seed (int): Random seed.
        steps (int): Number of optimisation steps.

    Returns:
        dict: Trial results including loss history, final distance, and timing.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark = benchmark_cls(dim)
    x = torch.tensor(
        np.random.uniform(INIT_RANGE[0], INIT_RANGE[1], size=dim),
        dtype=torch.float32, requires_grad=True
    )
    optimizer = _build_optimizer(opt_name, [x])

    losses = []
    t0 = time.perf_counter()

    for _ in range(steps):
        optimizer.zero_grad()
        loss = benchmark.evaluate(x)
        if torch.isnan(loss) or torch.isinf(loss):
            losses.append(float('nan'))
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], max_norm=10.0)
        optimizer.step()
        losses.append(loss.item())

    elapsed = time.perf_counter() - t0
    final_loss = losses[-1] if losses else float('nan')
    dist = benchmark.distance_to_global_min(x) if not np.isnan(final_loss) else float('nan')

    return {
        'seed':        seed,
        'optimizer':   opt_name,
        'benchmark':   benchmark.name,
        'dim':         dim,
        'losses':      losses,
        'final_loss':  final_loss,
        'dist_to_min': dist,
        'elapsed_s':   elapsed,
        'converged':   final_loss < 1e-3 if not np.isnan(final_loss) else False,
    }


def run_nn_trial(bench: NNLandmarkBenchmark, opt_name: str, seed: int,
                 epochs: int = 30) -> dict:
    """
    Run a single neural network landscape trial.

    Args:
        bench (NNLandmarkBenchmark): Pre-built benchmark instance.
        opt_name (str): Optimizer name.
        seed (int): Model initialisation seed.
        epochs (int): Number of training epochs.

    Returns:
        dict: Trial results including loss/accuracy history, Hessian trace, and timing.
    """
    model = bench.build_model(seed)
    optimizer = _build_optimizer(opt_name, model.parameters())

    n = len(bench.X)
    batch_size = min(bench.batch_size, n)

    train_losses, train_accs = [], []
    t0 = time.perf_counter()

    for epoch in range(epochs):
        # Shuffle data each epoch
        perm = torch.randperm(n)
        epoch_losses = []

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = bench.X[idx], bench.y[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = bench.criterion(logits, yb)
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        full_loss, full_acc = bench.compute_full_loss_and_acc(model)
        train_losses.append(full_loss)
        train_accs.append(full_acc)

    elapsed = time.perf_counter() - t0

    # Hessian trace at final parameters (basin sharpness)
    try:
        h_trace = bench.hessian_trace_approx(model, n_samples=3)
    except Exception:
        h_trace = float('nan')

    return {
        'seed':          seed,
        'optimizer':     opt_name,
        'benchmark':     bench.name,
        'n_params':      model.num_parameters(),
        'train_losses':  train_losses,
        'train_accs':    train_accs,
        'final_loss':    train_losses[-1] if train_losses else float('nan'),
        'final_acc':     train_accs[-1]   if train_accs   else float('nan'),
        'hessian_trace': h_trace,
        'elapsed_s':     elapsed,
    }


def run_dimensionality_sweep(opt_names=None, dims=None, seeds=None,
                              benchmarks=None, steps=STEPS) -> dict:
    """
    Run the dimensionality sweep experiment.

    For each (benchmark, optimizer, dimension, seed) combination, runs one
    mathematical benchmark trial and aggregates results.

    Returns:
        dict: Nested results keyed by benchmark -> optimizer -> dim -> list of trials.
    """
    opt_names  = opt_names  or list(OPTIMIZER_REGISTRY.keys())
    dims       = dims       or DIMENSIONALITY_SWEEP
    seeds      = seeds      or list(range(NUM_SEEDS))
    benchmarks = benchmarks or MATH_BENCHMARK_REGISTRY

    results = {}
    total = len(benchmarks) * len(opt_names) * len(dims) * len(seeds)
    done = 0

    for bname in benchmarks:
        results[bname] = {}
        bench_cls = BENCHMARK_MAP[bname]
        for opt in opt_names:
            results[bname][opt] = {}
            for dim in dims:
                results[bname][opt][dim] = []
                for seed in seeds:
                    trial = run_math_trial(bench_cls, dim, opt, seed, steps)
                    results[bname][opt][dim].append(trial)
                    done += 1
                    if done % 10 == 0:
                        print(f"  [{done}/{total}] {bname} | {opt} | dim={dim} | seed={seed} "
                              f"| loss={trial['final_loss']:.4f}")

    _save_results(results, 'dimensionality_sweep.json')
    return results


def run_nonconvexity_sweep(opt_names=None, alphas=None, dim=20,
                            seeds=None, steps=STEPS) -> dict:
    """
    Run the non-convexity dial experiment.

    Sweeps the perturbation amplitude alpha from 0 (convex) to large values
    (highly non-convex) on the NonConvexDial benchmark at a fixed dimensionality.

    Returns:
        dict: Nested results keyed by optimizer -> alpha -> list of trials.
    """
    opt_names = opt_names or list(OPTIMIZER_REGISTRY.keys())
    alphas    = alphas    or NONCONVEXITY_SWEEP
    seeds     = seeds     or list(range(NUM_SEEDS))

    results = {}
    total = len(opt_names) * len(alphas) * len(seeds)
    done = 0

    for opt in opt_names:
        results[opt] = {}
        for alpha in alphas:
            results[opt][alpha] = []
            for seed in seeds:
                bench = NonConvexDial(dim=dim, alpha=alpha)
                trial = run_math_trial(NonConvexDial, dim, opt, seed, steps)
                # Override with the correct alpha-specific benchmark
                torch.manual_seed(seed)
                np.random.seed(seed)
                x = torch.tensor(
                    np.random.uniform(INIT_RANGE[0], INIT_RANGE[1], size=dim),
                    dtype=torch.float32, requires_grad=True
                )
                optimizer = _build_optimizer(opt, [x])
                losses = []
                for _ in range(steps):
                    optimizer.zero_grad()
                    loss = bench.evaluate(x)
                    if torch.isnan(loss) or torch.isinf(loss):
                        losses.append(float('nan'))
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([x], max_norm=10.0)
                    optimizer.step()
                    losses.append(loss.item())
                final_loss = losses[-1] if losses else float('nan')
                results[opt][alpha].append({
                    'seed': seed, 'alpha': alpha, 'losses': losses,
                    'final_loss': final_loss,
                    'converged': final_loss < 0.5 if not np.isnan(final_loss) else False,
                })
                done += 1
                if done % 5 == 0:
                    print(f"  [{done}/{total}] {opt} | alpha={alpha} | seed={seed} "
                          f"| loss={final_loss:.4f}")

    _save_results(results, 'nonconvexity_sweep.json')
    return results


def run_nn_suite(opt_names=None, seeds=None) -> dict:
    """
    Run all neural network landscape benchmarks.

    Returns:
        dict: Nested results keyed by benchmark_name -> optimizer -> list of trials.
    """
    opt_names = opt_names or list(OPTIMIZER_REGISTRY.keys())
    seeds     = seeds     or list(range(min(NUM_SEEDS, 5)))  # fewer seeds for NN (slower)

    results = {}
    for cfg in NN_BENCHMARK_REGISTRY:
        bname = cfg['name']
        print(f"\n  Building benchmark: {bname} (hidden={cfg['hidden']})")
        bench = NNLandmarkBenchmark(
            name=bname,
            dataset=cfg['dataset'],
            hidden_sizes=cfg['hidden'],
        )
        results[bname] = {}
        for opt in opt_names:
            results[bname][opt] = []
            for seed in seeds:
                print(f"    {opt} | seed={seed}", end=' ')
                trial = run_nn_trial(bench, opt, seed, epochs=cfg['epochs'])
                results[bname][opt].append(trial)
                print(f"| loss={trial['final_loss']:.4f} acc={trial['final_acc']:.1f}%")

    _save_results(results, 'nn_suite.json')
    return results


def _save_results(results: dict, filename: str):
    """Serialise results to JSON, converting non-serialisable types."""
    def _convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    path = RESULTS_DIR / filename
    with open(path, 'w') as f:
        json.dump(_convert(results), f, indent=2)
    print(f"\n  Results saved to {path}")
