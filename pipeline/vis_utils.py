import matplotlib.pyplot as plt
import os
import json


import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
def plot_training_curves(history, save_dir, exp_name):
    """
    Строит графики лосса, метрики MAE и Learning Rate.
    history: dict {'train_loss': [], 'val_loss': [], 'val_mae': [], 'lr': []}
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Training Metrics: {exp_name}", fontsize=16)

    # 1. Loss Curves (MSLE)
    axes[0].plot(epochs, history['train_loss'], label='Train MSLE', marker='.')
    axes[0].plot(epochs, history['val_loss'], label='Val MSLE', marker='.')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss (Log space)')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True)

    # 2. Metric Curves (Real MAE)
    axes[1].plot(epochs, history['val_mae_Z1'], label='Val Real MAE', color='orange', marker='.')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('MAE (Concentration Units)')
    axes[1].set_title('Validation MAE Z1 (Physical)')
    axes[1].legend()
    axes[1].grid(True)

    # 3. Learning Rate Schedule
    axes[2].plot(epochs, history['lr'], label='Learning Rate', color='green', linestyle='--')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('LR')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True)

    plt.tight_layout()
    
    # Сохранение графика
    plot_path = os.path.join(save_dir, f'{exp_name}_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Сохранение сырых данных (json), чтобы можно было перестроить график позже
    json_path = os.path.join(save_dir, f'{exp_name}_history.json')
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4, cls=NumpyEncoder)
        
    return plot_path