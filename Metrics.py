"""
Metrics calculation utilities.
"""
import numpy as np

def calculate_benchmark_metrics(final_point, global_min, path, loss_values):
    """
    Calculate metrics for a benchmark optimization run.
    
    Args:
        final_point (np.ndarray): The final optimized point.
        global_min (np.ndarray): The known global minimum.
        path (list): The path of points taken during optimization.
        loss_values (list): The loss values recorded during optimization.
        
    Returns:
        dict: Distance to global min, final loss, and convergence rate.
    """
    distance = np.linalg.norm(np.array(final_point) - np.array(global_min))
    convergence_rate = len(path) if loss_values[-1] < 1e-5 else float('inf')
    return {
        'distance': float(distance),
        'final_loss': float(loss_values[-1]),
        'convergence_rate': convergence_rate
    }

def calculate_ml_metrics(train_history, val_history):
    """
    Calculate metrics for an ML training run.
    
    Args:
        train_history (dict): Training history (loss, accuracy).
        val_history (dict): Validation history (loss, accuracy).
        
    Returns:
        dict: Final accuracies, generalization gap, final losses, and best epoch.
    """
    final_train_acc = train_history['accuracy'][-1]
    final_val_acc = val_history['accuracy'][-1]
    generalization_gap = final_train_acc - final_val_acc
    final_train_loss = train_history['loss'][-1]
    final_val_loss = val_history['loss'][-1]
    best_epoch = int(np.argmax(val_history['accuracy']) + 1)
    
    return {
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'generalization_gap': generalization_gap,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_epoch': best_epoch
    }
