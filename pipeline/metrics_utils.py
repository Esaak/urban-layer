
import torch
import numpy as np

class AirQualityMetrics:
    def __init__(self, layer_names=None):
        self.layer_names = layer_names or ["Z1", "Z5", "Z8", "Z25", "Mean"]
        self.num_layers = len(self.layer_names)
        self.reset()

    def reset(self):
        # Списки для хранения предсказаний и таргетов по каждому слою
        self.obs_per_layer = [[] for _ in range(self.num_layers)]
        self.pred_per_layer = [[] for _ in range(self.num_layers)]

    def update(self, pred, target):
        """
        pred, target: тензоры формы [B, C, H, W] в реальных величинах
        """
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        for i in range(self.num_layers):
            # Извлекаем i-й канал и выпрямляем его
            p = pred_np[:, i, :, :].flatten()
            t = target_np[:, i, :, :].flatten()
            
            self.obs_per_layer[i].append(t)
            self.pred_per_layer[i].append(p)

    def compute(self):
        overall_metrics = {}
        layer_metrics = {}

        all_obs_flat = []
        all_pred_flat = []

        for i in range(self.num_layers):
            name = self.layer_names[i]
            obs = np.concatenate(self.obs_per_layer[i])
            pred = np.concatenate(self.pred_per_layer[i])
            
            # Считаем метрики для конкретного слоя
            m = self._calculate_stats(obs, pred)
            layer_metrics[name] = m
            
            all_obs_flat.append(obs)
            all_pred_flat.append(pred)

        # Считаем общую метрику (по всем слоям)
        overall_metrics = self._calculate_stats(
            np.concatenate(all_obs_flat), 
            np.concatenate(all_pred_flat)
        )

        return overall_metrics, layer_metrics

    def _calculate_stats(self, obs, pred):
        """Вспомогательная функция для расчета базы"""
        mae = np.mean(np.abs(obs - pred))
        
        # Fractional Bias
        mean_obs = np.mean(obs)
        mean_pred = np.mean(pred)
        fb = 2 * (mean_obs - mean_pred) / (mean_obs + mean_pred + 1e-8)
        
        # R2
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - mean_obs) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {'mae': mae, 'fb': fb, 'r2': r2}