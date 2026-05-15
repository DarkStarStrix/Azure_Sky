"""
Main entry point for the AzureSky benchmarking suite.

Usage:
    python benchmarking/run_suite.py [--quick] [--math-only] [--nn-only]

Flags:
    --quick      Run a reduced config (fewer seeds, fewer dims) for fast validation.
    --math-only  Skip neural network benchmarks.
    --nn-only    Skip mathematical benchmarks.
"""
import sys
import argparse
from pathlib import Path

# Ensure repo root is on the path when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.runners.suite_runner import (
    run_dimensionality_sweep,
    run_nonconvexity_sweep,
    run_nn_suite,
)
from benchmarking.reporters.report_generator import (
    plot_convergence_curves,
    plot_dimensionality_scaling,
    plot_nonconvexity_dial,
    plot_reliability,
    plot_nn_training_curves,
    plot_basin_sharpness,
    generate_summary_table,
)
from benchmarking.configs.suite_config import (
    DIMENSIONALITY_SWEEP, NONCONVEXITY_SWEEP, MATH_BENCHMARK_REGISTRY,
    NN_BENCHMARK_REGISTRY, NUM_SEEDS, OPTIMIZER_REGISTRY,
)


def main():
    parser = argparse.ArgumentParser(description='AzureSky Benchmarking Suite')
    parser.add_argument('--quick',     action='store_true',
                        help='Run a reduced config for fast validation')
    parser.add_argument('--math-only', action='store_true',
                        help='Skip neural network benchmarks')
    parser.add_argument('--nn-only',   action='store_true',
                        help='Skip mathematical benchmarks')
    args = parser.parse_args()

    # ── Config overrides for quick mode ──────────────────────────────────────
    if args.quick:
        dims       = [2, 10, 20]
        alphas     = [0.0, 1.0, 5.0]
        seeds      = list(range(3))
        steps      = 200
        benchmarks = ['Ackley', 'Rastrigin', 'Rosenbrock']
        nn_configs = [c for c in NN_BENCHMARK_REGISTRY
                      if c['name'] in ('TwoMoons_Small', 'TwoMoons_Medium')]
        nn_epochs  = 15
        print("Running in QUICK mode (reduced seeds, dims, and steps)")
    else:
        dims       = DIMENSIONALITY_SWEEP
        alphas     = NONCONVEXITY_SWEEP
        seeds      = list(range(NUM_SEEDS))
        steps      = 500
        benchmarks = MATH_BENCHMARK_REGISTRY
        nn_configs = NN_BENCHMARK_REGISTRY
        nn_epochs  = None  # use per-config default
        print("Running in FULL mode")

    opt_names = list(OPTIMIZER_REGISTRY.keys())
    print(f"Optimizers : {opt_names}")
    print(f"Seeds      : {len(seeds)}")
    print(f"Steps      : {steps}")
    print()

    dim_results = {}
    nc_results  = {}
    nn_results  = {}

    # ── Mathematical benchmarks ───────────────────────────────────────────────
    if not args.nn_only:
        print("=" * 60)
        print("PHASE 1: Dimensionality Sweep")
        print("=" * 60)
        dim_results = run_dimensionality_sweep(
            opt_names=opt_names, dims=dims, seeds=seeds,
            benchmarks=benchmarks, steps=steps
        )

        print("\n" + "=" * 60)
        print("PHASE 2: Non-Convexity Dial")
        print("=" * 60)
        nc_results = run_nonconvexity_sweep(
            opt_names=opt_names, alphas=alphas, dim=20,
            seeds=seeds, steps=steps
        )

    # ── Neural network benchmarks ─────────────────────────────────────────────
    if not args.math_only:
        print("\n" + "=" * 60)
        print("PHASE 3: Neural Network Landscape Suite")
        print("=" * 60)
        # Temporarily override NN_BENCHMARK_REGISTRY epochs if quick mode
        if args.quick:
            import benchmarking.configs.suite_config as cfg
            original = cfg.NN_BENCHMARK_REGISTRY
            cfg.NN_BENCHMARK_REGISTRY = [
                {**c, 'epochs': nn_epochs} for c in nn_configs
            ]
        nn_results = run_nn_suite(opt_names=opt_names, seeds=seeds)
        if args.quick:
            cfg.NN_BENCHMARK_REGISTRY = original

    # ── Reporting ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: Generating Reports")
    print("=" * 60)

    if dim_results:
        print("\n  Convergence curves...")
        plot_convergence_curves(dim_results, dim=20, benchmarks=benchmarks)

        print("  Dimensionality scaling...")
        plot_dimensionality_scaling(dim_results, benchmarks=benchmarks)

        print("  Reliability heatmap...")
        plot_reliability(dim_results, benchmarks=benchmarks, dims=[2, 10])

    if nc_results:
        print("  Non-convexity dial...")
        plot_nonconvexity_dial(nc_results)

    if nn_results:
        print("  NN training curves...")
        plot_nn_training_curves(nn_results)

        print("  Basin sharpness...")
        plot_basin_sharpness(nn_results)

    # ── Summary table ─────────────────────────────────────────────────────────
    if dim_results and nc_results and nn_results:
        print("\n  Summary table:")
        table = generate_summary_table(dim_results, nc_results, nn_results)
        print("\n" + table)
        summary_path = Path(__file__).parent / 'results' / 'summary_table.md'
        summary_path.write_text(table)
        print(f"\n  Saved to {summary_path}")

    print("\nDone. All plots saved to docs/images/benchmarking/")


if __name__ == '__main__':
    main()
