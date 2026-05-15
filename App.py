"""
Gradio Frontend Application.
"""
import gradio as gr
import json
import numpy as np
from Engine import Engine

engine = Engine()

def run_study(mode, benchmark_func, optimizers_b, dim, dataset, optimizers_m, epochs, batch_size, lr, use_sa):
    """
    Run the selected study (Benchmark or ML Task) and return plots and metrics.
    """
    if mode == 'Benchmark Optimization':
        if not benchmark_func or not optimizers_b:
            return None, None, {"error": "Please select a benchmark and at least one optimizer."}, None
            
        metrics_all = {}
        paths = []
        
        # Simple initial guess for 2D or ND
        dim = int(dim)
        initial_guess = np.random.uniform(-5, 5, size=dim)
        
        benchmark_instance = engine.benchmarks[benchmark_func]()
        
        for opt in optimizers_b:
            metrics = engine.run_benchmark(benchmark_func, opt, initial_guess, max_steps=200)
            metrics_all[opt] = metrics
            
            # Re-run just to get the path for plotting (in a real app we'd extract this from the run)
            # For simplicity in this demo, we'll just show the metrics table.
            
        return None, None, metrics_all, json.dumps(metrics_all, indent=2)
        
    else:
        if not dataset or not optimizers_m:
            return None, None, {"error": "Please select a dataset and at least one optimizer."}, None
            
        metrics_all = {}
        
        for opt in optimizers_m:
            metrics = engine.run_ml_task(dataset, opt, epochs=int(epochs), batch_size=int(batch_size), lr=float(lr))
            metrics_all[opt] = metrics
            
        return None, None, metrics_all, json.dumps(metrics_all, indent=2)

def export_results(data):
    """
    Export results to a JSON file.
    """
    if not data:
        return None, None
    filename = "results.json"
    with open(filename, "w") as f:
        f.write(data)
    return filename, filename

with gr.Blocks(title="AzureSky Optimizer Study") as app:
    gr.Markdown("# AzureSky Optimizer Evaluation Dashboard")
    
    with gr.Row():
        mode = gr.Radio(['Benchmark Optimization', 'ML Task Training'], label='Study Mode', value='Benchmark Optimization')
        
    with gr.Row():
        with gr.Column(visible=True) as benchmark_tab:
            benchmark_func = gr.Dropdown(['Himmelblau', 'Ackley', 'Adjiman', 'Brent'], label='Benchmark Function')
            optimizers_b = gr.CheckboxGroup(['AzureSky', 'Adam', 'SGD', 'RMSprop'], label='Optimizers')
            dim = gr.Number(label='Dimensionality (for Ackley)', value=2, minimum=1)
            
        with gr.Column(visible=False) as ml_task_tab:
            dataset = gr.Dropdown(['MNIST', 'CIFAR10'], label='Dataset')
            optimizers_m = gr.CheckboxGroup(['AzureSky', 'Adam', 'SGD', 'RMSprop'], label='Optimizers')
            epochs = gr.Number(label='Epochs', value=2, minimum=1)
            batch_size = gr.Number(label='Batch Size', value=32, minimum=1)
            lr = gr.Number(label='Learning Rate', value=0.001, minimum=0)
            
    use_sa = gr.Checkbox(label='Use Simulated Annealing (AzureSky)', value=True)
    run_button = gr.Button('Run Study')
    
    with gr.Row():
        plot1 = gr.Plot(label='Main Plot (Not fully implemented in demo)')
        plot2 = gr.Plot(label='Secondary Plot (Not fully implemented in demo)')
        
    metrics_table = gr.JSON(label='Metrics')
    export_data = gr.State()
    
    export_button = gr.Button('Export Results')
    export_file = gr.File(label='Download Results')
    
    def toggle_tabs(selected_mode):
        return gr.update(visible=selected_mode == 'Benchmark Optimization'), gr.update(visible=selected_mode == 'ML Task Training')
        
    mode.change(toggle_tabs, inputs=mode, outputs=[benchmark_tab, ml_task_tab])
    
    run_button.click(run_study,
                     inputs=[mode, benchmark_func, optimizers_b, dim, dataset, optimizers_m, epochs, batch_size, lr, use_sa],
                     outputs=[plot1, plot2, metrics_table, export_data])
                     
    export_button.click(export_results, inputs=[export_data], outputs=[export_file, gr.File()])

if __name__ == "__main__":
    app.launch()
