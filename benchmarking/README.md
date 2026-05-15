# AzureSky Benchmarking Suite

A comprehensive, fair, and extensible benchmarking suite for evaluating the AzureSky optimizer against Adam, AdamW, SGD, and RMSprop across synthetic mathematical landscapes, neural network loss surfaces, and controlled non-convexity experiments.

## Fairness Guarantee

All optimizers share identical hyperparameters wherever a parameter is common: `lr=0.01`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`. AzureSky's SA-specific parameters (`sa_steps`, `T0`, `sigma`) have no equivalent in other optimizers and are documented separately in `configs/suite_config.py`. No optimizer receives a tuned advantage.

## Suite Structure

```
benchmarking/
├── configs/
│   └── suite_config.py        # Central config: optimizer registry, sweep params
├── benchmarks/
│   ├── math_functions.py      # Ackley, Rastrigin, Rosenbrock, Schwefel, NonConvexDial
│   └── nn_landscapes.py       # MLP on Two Moons / Swiss Roll + Hessian trace
├── runners/
│   └── suite_runner.py        # Dimensionality sweep, NC dial, NN suite runners
├── reporters/
│   └── report_generator.py    # All plot functions + summary table
├── results/                   # JSON result files (auto-generated)
└── run_suite.py               # Main entry point
```

## Running the Suite

```bash
# Fast validation (3 seeds, 3 dims, 200 steps)
python benchmarking/run_suite.py --quick

# Full suite (10 seeds, 6 dims, 500 steps)
python benchmarking/run_suite.py

# Math benchmarks only
python benchmarking/run_suite.py --math-only

# NN benchmarks only
python benchmarking/run_suite.py --nn-only
```

## Experiments

| Experiment | What it measures |
|---|---|
| **Dimensionality Sweep** | Final loss and convergence curves for Ackley, Rastrigin, Rosenbrock, Schwefel at dims 2→100 |
| **Non-Convexity Dial** | How each optimizer degrades as perturbation amplitude α increases from 0 (convex) to 10 (highly non-convex) |
| **NN Landscape Suite** | Training loss, accuracy, and Hessian trace (basin sharpness) on Two Moons and Swiss Roll at 3 model scales |
| **Reliability Heatmap** | % of trials landing in the bottom-25th percentile loss across seeds and dims |

## Plots Generated

All plots are saved to `docs/images/benchmarking/`:

- `convergence_{benchmark}_dim20.png` — median ± IQR convergence curves
- `dim_scaling_{benchmark}.png` — final loss vs dimensionality
- `nonconvexity_dial.png` — final loss vs α
- `reliability_heatmap.png` — convergence reliability heatmap
- `nn_{benchmark}.png` — NN training loss and accuracy curves
- `basin_sharpness.png` — Hutchinson Hessian trace at convergence

## Extending the Suite

**Add an optimizer:** Add an entry to `OPTIMIZER_REGISTRY` in `configs/suite_config.py`.

**Add a benchmark:** Subclass `ScalableBenchmark` in `benchmarks/math_functions.py` and add the name to `MATH_BENCHMARK_REGISTRY`.

**Change scale:** Edit `DIMENSIONALITY_SWEEP`, `NONCONVEXITY_SWEEP`, `NUM_SEEDS`, or `STEPS` in `configs/suite_config.py`.

**Add an NN task:** Add an entry to `NN_BENCHMARK_REGISTRY` with a `dataset`, `hidden`, and `epochs` field.
