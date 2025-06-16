import gradio as gr
import json
from Engine import Engine

def run_study(mode, benchmark_func, optimizers, dim, dataset, epochs, batch_size, lr, use_sa):
    config = {
        'mode': 'benchmark' if mode == 'Benchmark Optimization' else 'ml_task',
        'benchmark_func': benchmark_func,
        'optimizers': optimizers,
        'dim': dim,
        'dataset': dataset,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'use_sa': use_sa if 'AzureSky' in optimizers else None,
        'max_iter': 100  # Default for benchmark mode
    }
    runner = Engine()
    results = runner.run()

    if config['mode'] == 'benchmark':
        return results['plot'], None, results['metrics'], json.dumps(results)
    else:
        return results['plot_acc'], results['plot_loss'], results['metrics'], json.dumps(results)

def export_results(results_json):
    return results_json, "results.json"

with gr.Blocks(title="Nexa R&D Studio") as app:
    mode = gr.Radio(['Benchmark Optimization', 'ML Task Training'], label='Mode', value='Benchmark Optimization')

    with gr.Row():
        with gr.Column(visible=True) as benchmark_tab:
            benchmark_func = gr.Dropdown(['Himmelblau', 'Ackley', 'Adjiman','Brent'], label='Benchmark Function')
            optimizers_b = gr.CheckboxGroup(['AzureSky', 'Adam', 'SGD'], label='Optimizers')
            dim = gr.Number(label='Dimensionality', value=2, minimum=2)
        with gr.Column(visible=False) as ml_task_tab:
            dataset = gr.Dropdown(['MNIST', 'CIFAR-10'], label='Dataset')
            optimizers_m = gr.CheckboxGroup(['AzureSky', 'Adam', 'SGD'], label='Optimizers')
            epochs = gr.Number(label='Epochs', value=10, minimum=1)
            batch_size = gr.Number(label='Batch Size', value=32, minimum=1)
            lr = gr.Number(label='Learning Rate', value=0.001, minimum=0)

    use_sa = gr.Checkbox(label='Use Simulated Annealing (AzureSky)', value=True)
    run_button = gr.Button('Run Study')

    plot1 = gr.Plot(label='Main Plot')
    plot2 = gr.Plot(label='Secondary Plot (ML Mode)')
    metrics_table = gr.JSON(label='Metrics')
    export_data = gr.State()
    export_button = gr.Button('Export Results')
    export_file = gr.File(label='Download Results')

    def toggle_tabs(mode):
        return gr.update(visible=mode == 'Benchmark Optimization'), gr.update(visible=mode == 'ML Task Training')

    mode.change(toggle_tabs, inputs=mode, outputs=[benchmark_tab, ml_task_tab])
    run_button.click(run_study,
                     inputs=[mode, benchmark_func, optimizers_b, dim, dataset, epochs, batch_size, lr, use_sa],
                     outputs=[plot1, plot2, metrics_table, export_data])
    export_button.click(export_results, inputs=[export_data], outputs=[export_file, gr.File()])

app.launch()
