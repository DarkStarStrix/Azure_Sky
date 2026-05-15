"""
Load the saved himmelblau_scale.json and regenerate all plots.
Run this after the sweep has completed to avoid re-running the expensive computation.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_PATH = Path(__file__).parent / 'results' / 'himmelblau_scale.json'

with open(RESULTS_PATH) as f:
    raw = json.load(f)

# Convert string keys back to int keys
results = {}
for opt, dim_map in raw.items():
    results[opt] = {int(k): v for k, v in dim_map.items()}

# Import all plot functions from the main script
from benchmarking.run_himmelblau_scale import (
    plot_convergence_grid,
    plot_final_loss_scaling,
    plot_distance_to_min_scaling,
    plot_convergence_speed,
    plot_reliability_heatmap,
    plot_variance_analysis,
    plot_summary_panel,
)

print("Generating all Himmelblau plots from saved results...")
plot_convergence_grid(results)
plot_final_loss_scaling(results)
plot_distance_to_min_scaling(results)
plot_convergence_speed(results)
plot_reliability_heatmap(results)
plot_variance_analysis(results)
plot_summary_panel(results)
print("Done. All plots saved to docs/images/benchmarking/")
