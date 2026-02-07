import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
from model import UNet25D

def get_latest_exp_dir(base_dir='./outputs/'):
    folders = sorted(glob.glob(os.path.join(base_dir, 'output_*')))
    if not folders: raise FileNotFoundError("No experiments found.")
    return folders[-1]

def run_inference(data_dir, device='cuda'):
    # 1. Загрузка последнего эксперимента
    exp_dir = get_latest_exp_dir()
    model_path = glob.glob(os.path.join(exp_dir, '*_best.pth'))[0]
    
    model = UNet25D(n_channels=7, n_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Выбор данных
    sample_path = random.choice(glob.glob(os.path.join(data_dir, '*.npz')))
    data = np.load(sample_path)
    x, y_gt_log = data['x'], data['y']
    
    # 3. Предсказание
    x_tensor = torch.from_numpy(x).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_log = model(x_tensor)
    
    pred_real = torch.clamp(torch.expm1(pred_log), min=0).squeeze().cpu().numpy()
    #y_gt_real = np.expm1(y_gt_log)

    # 4. Визуализация (упрощенно для примера, можно использовать ваш код 5x4)
    fig, axes = plt.subplots(5, 4, figsize=(15, 20))
    layers = ["Ground", "Z5", "Roofs", "High", "Mean"]
    
    for i in range(5):
        vmax = max(y_gt_log[i].max(), pred_real[i].max())
        im1 = axes[i, 1].imshow(y_gt_log[i], cmap='turbo', origin='lower', vmax=vmax)
        axes[i, 1].set_title(f"GT {layers[i]}")
        plt.colorbar(im1, ax=axes[i,1], fraction=0.046, pad=0.04)

        im2 = axes[i, 2].imshow(pred_real[i], cmap='turbo', origin='lower', vmax=vmax)
        axes[i, 2].set_title(f"Pred {layers[i]}")
        plt.colorbar(im2, ax=axes[i,2], fraction=0.046, pad=0.04)

        err = np.abs(y_gt_log[i] - pred_real[i])
        im = axes[i, 3].imshow(err, cmap='magma', origin='lower')
        plt.colorbar(im, ax=axes[i, 3])
        axes[i, 3].set_title(f"Error (Max: {err.max():.2f})")
        
        if i == 0:
            im0 = axes[i,0].imshow(x[0], cmap='bone', origin='lower')
            axes[i,0].set_title("Input: BuildH")
        elif i == 1:
            im0 = axes[i,0].imshow(x[1], cmap='viridis', origin='lower')
            axes[i,0].set_title("Input: SDF")
        elif i == 2:
            im0 = axes[i,0].imshow(x[3], cmap='hot', origin='lower')
            axes[i,0].set_title("Input: SourceH")
        elif i == 3:
            # Направление ветра (векторное поле)
            axes[i,0].set_title(f"Wind: DX={x[4,0,0]:.5f}, DY={x[5,0,0]:.5f}")
            axes[i,0].axis('off')
        else:
            im0 = axes[i,0].imshow(x[2], cmap='Greens', origin='lower')
            axes[i,0].set_title("Input: LAI (Trees)")
        
        if i != 3: plt.colorbar(im0, ax=axes[i,0], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_name = os.path.join(exp_dir, f"inf_{os.path.basename(sample_path)}.png")
    plt.savefig(save_name)
    print(f"Result saved to {save_name}")

if __name__ == "__main__":
    run_inference('/app/urban-layer-datasets/2026_01_19_500_25d_data_1/')