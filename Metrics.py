import numpy as np

def calculate_benchmark_metrics(final_point, global_min, path, loss_values):
    distance = np.linalg.norm(final_point - global_min)
    convergence_rate = len(path) if loss_values[-1] < 1e-5 else float('inf')
    return {'distance': float(distance), 'final_loss': float(loss_values[-1]), 'convergence_rate': convergence_rate}

def calculate_ml_metrics(train_history, val_history):
    final_train_acc = train_history['accuracy'][-1]
    final_val_acc = val_history['accuracy'][-1]
    generalization_gap = final_train_acc - final_val_acc
    final_train_loss = train_history['loss'][-1]
    final_val_loss = val_history['loss'][-1]
    best_epoch = np.argmax(val_history['accuracy']) + 1
    return {
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'generalization_gap': generalization_gap,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_epoch': best_epoch
    }
