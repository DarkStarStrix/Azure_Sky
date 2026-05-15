"""
Central configuration for the AzureSky benchmarking suite.

All optimizers are given identical shared hyperparameters wherever a parameter
is common to all of them. Optimizer-specific parameters (e.g. sa_steps for
AzureSky) are listed separately and clearly documented. This ensures that no
optimizer receives an unfair advantage through tuning.

To add a new optimizer: add an entry to OPTIMIZER_REGISTRY.
To add a new benchmark: add an entry to MATH_BENCHMARK_REGISTRY or NN_BENCHMARK_REGISTRY.
To change scale: edit DIMENSIONALITY_SWEEP or NONCONVEXITY_SWEEP.
"""

# ── Shared hyperparameters (identical across all optimizers) ─────────────────
SHARED_LR         = 0.01
SHARED_MOMENTUM   = 0.9    # used by SGD; ignored by Adam-family
SHARED_BETA1      = 0.9    # used by Adam-family; ignored by SGD/RMSprop
SHARED_BETA2      = 0.999  # used by Adam-family; ignored by SGD/RMSprop
SHARED_EPS        = 1e-8
SHARED_WEIGHT_DECAY = 0.0  # zero for all — no regularisation advantage

# ── AzureSky-specific parameters (documented explicitly) ─────────────────────
# These have no equivalent in other optimizers and cannot be "shared".
# sa_steps controls how long the SA phase is active; T0 and sigma control
# the temperature and perturbation magnitude of the SA phase.
AZURE_SA_STEPS    = 300
AZURE_T0          = 1.0
AZURE_SIGMA       = 0.1
AZURE_SA_MOMENTUM = 0.9

# ── Optimizer registry ────────────────────────────────────────────────────────
# Each entry: (import_path, class_name, kwargs)
# kwargs must only contain parameters that are valid for that optimizer.
OPTIMIZER_REGISTRY = {
    'AzureSky': {
        'module': 'Backend.optimizers.azure_optim',
        'class':  'Azure',
        # AzureSky shares lr, betas, eps with Adam-family.
        # sa_steps, T0, sigma, sa_momentum are SA-specific and have no
        # equivalent in other optimizers — they cannot be "shared".
        'kwargs': {
            'lr':          SHARED_LR,
            'betas':       (SHARED_BETA1, SHARED_BETA2),
            'eps':         SHARED_EPS,
            'sa_steps':    AZURE_SA_STEPS,
            'T0':          AZURE_T0,
            'sigma':       AZURE_SIGMA,
            'sa_momentum': AZURE_SA_MOMENTUM,
        },
    },
    'Adam': {
        'module': 'Backend.optimizers.adam',
        'class':  'AdamOptimizer',
        'kwargs': {
            'lr':    SHARED_LR,
            'betas': (SHARED_BETA1, SHARED_BETA2),
            'eps':   SHARED_EPS,
        },
    },
    'AdamW': {
        'module': 'Backend.optimizers.adamw',
        'class':  'AdamWOptimizer',
        'kwargs': {
            'lr':           SHARED_LR,
            'betas':        (SHARED_BETA1, SHARED_BETA2),
            'eps':          SHARED_EPS,
            'weight_decay': SHARED_WEIGHT_DECAY,
        },
    },
    'SGD': {
        'module': 'Backend.optimizers.SGD',
        'class':  'SGDOptimizer',
        'kwargs': {
            'lr':       SHARED_LR,
            'momentum': SHARED_MOMENTUM,
        },
    },
    'RMSprop': {
        'module': 'Backend.optimizers.RMSprop',
        'class':  'RMSpropOptimizer',
        'kwargs': {
            'lr':  SHARED_LR,
            'eps': SHARED_EPS,
        },
    },
}

# ── Trial settings ────────────────────────────────────────────────────────────
NUM_SEEDS    = 10    # number of independent random seeds per (optimizer, benchmark) pair
STEPS        = 500   # optimisation steps per trial
INIT_RANGE   = (-3.0, 3.0)  # uniform range for initial parameter sampling

# ── Dimensionality sweep ──────────────────────────────────────────────────────
DIMENSIONALITY_SWEEP = [2, 5, 10, 20, 50, 100]

# ── Non-convexity dial ────────────────────────────────────────────────────────
# Amplitude of the sinusoidal perturbation added to a base quadratic.
# 0.0 = perfectly convex; higher values = increasingly non-convex.
NONCONVEXITY_SWEEP = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

# ── Mathematical benchmark registry ──────────────────────────────────────────
MATH_BENCHMARK_REGISTRY = ['Ackley', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'Himmelblau']

# ── Neural network benchmark registry ────────────────────────────────────────
# Each entry: (dataset, hidden_sizes) — controls effective parameter-space dimensionality
NN_BENCHMARK_REGISTRY = [
    {'name': 'TwoMoons_Small',  'dataset': 'two_moons', 'hidden': [32, 32],         'epochs': 30},
    {'name': 'TwoMoons_Medium', 'dataset': 'two_moons', 'hidden': [128, 128],        'epochs': 30},
    {'name': 'TwoMoons_Large',  'dataset': 'two_moons', 'hidden': [256, 256, 256],   'epochs': 30},
    {'name': 'SwissRoll_Small', 'dataset': 'swiss_roll', 'hidden': [64, 64],         'epochs': 30},
    {'name': 'SwissRoll_Large', 'dataset': 'swiss_roll', 'hidden': [256, 256, 256],  'epochs': 30},
]
