import torch
import numpy as np

class AirQualityMetrics:
    """Класс для расчета специфичных метрик качества воздуха."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.abs_error = 0.0
        self.sq_error = 0.0
        self.sum_obs = 0.0
        self.sum_pred = 0.0
        self.n_samples = 0
        
        # Для R2
        self.all_obs = []
        self.all_pred = []

    def update(self, pred, target):
        """
        Обновляет статистику на батче.
        pred, target: тензоры в реальных (не лог) величинах.
        """
        pred = pred.detach().cpu().numpy().flatten()
        target = target.detach().cpu().numpy().flatten()

        self.abs_error += np.sum(np.abs(pred - target))
        self.sq_error += np.sum((pred - target)**2)
        self.sum_obs += np.sum(target)
        self.sum_pred += np.sum(pred)
        self.n_samples += len(target)
        
        # Сохраняем для корректного R2 (по всему набору)
        self.all_obs.append(target)
        self.all_pred.append(pred)

    def compute(self):
        if self.n_samples == 0:
            return {}

        obs_flat = np.concatenate(self.all_obs)
        pred_flat = np.concatenate(self.all_pred)

        # 1. MAE
        mae = self.abs_error / self.n_samples

        # 2. Fractional Bias (FB)
        # FB = 2 * (mean_obs - mean_pred) / (mean_obs + mean_pred)
        mean_obs = np.mean(obs_flat)
        mean_pred = np.mean(pred_flat)
        fb = 2 * (mean_obs - mean_pred) / (mean_obs + mean_pred + 1e-8)

        # 3. R-squared (R2)
        ss_res = np.sum((obs_flat - pred_flat) ** 2)
        ss_tot = np.sum((obs_flat - mean_obs) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'mae': mae,
            'fb': fb,
            'r2': r2
        }