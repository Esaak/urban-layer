import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import random

class UrbanAirDataset(Dataset):
    def __init__(self, data_dir, mode='train', log_target=True):
        """
        Args:
            data_dir: Путь к папке с .npz
            mode: 'train' (80% + аугментации) или 'val' (20% без аугментаций)
            log_target: Применять ли log1p к концентрации (рекомендуется True)
        """
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        
        # Детерминированное разбиение
        random.seed(42)
        random.shuffle(self.files)
        split_idx = int(len(self.files) * 0.8)
        
        if mode == 'train':
            self.files = self.files[:split_idx]
            self.transform = True
        else:
            self.files = self.files[split_idx:]
            self.transform = False
            
        self.log_target = log_target

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        
        # x shape: (6, 128, 128) -> [BuildH, SDF, LAI, SourceH, WindX, WindY]
        x = data['x'].astype(np.float32)
        # y shape: (5, 128, 128) -> [Z1, Z5, Z8, Z25, Mean]
        y = data['y'].astype(np.float32)
        
        # Аугментация (только для Train)
        if self.transform:
            x, y = self._augment(x, y)

        # Препроцессинг таргета (Log-space для стабилизации градиентов)
        if self.log_target:
            y = np.log1p(y)
            
        return torch.from_numpy(x), torch.from_numpy(y)

    def _augment(self, x, y):
        # 1. Случайный поворот (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            # Вращаем пространственные оси (H, W) -> последние две (1, 2)
            x = np.rot90(x, k, axes=(1, 2)).copy()
            y = np.rot90(y, k, axes=(1, 2)).copy()
            
            # Коррекция вектора ветра (каналы 4 и 5)
            # Внимание: np.rot90 вращает против часовой стрелки (Counter-Clockwise)
            u = x[4].copy()
            v = x[5].copy()
            
            if k == 1:   # 90 deg CCW: (x, y) -> (-y, x)
                x[4] = -v
                x[5] = u
            elif k == 2: # 180 deg: (x, y) -> (-x, -y)
                x[4] = -u
                x[5] = -v
            elif k == 3: # 270 deg CCW: (x, y) -> (y, -x)
                x[4] = v
                x[5] = -u

        # 2. Случайное отражение (Flip Left-Right)
        if random.random() > 0.5:
            # Flip по последней оси (Width / X)
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=2).copy()
            
            # При отражении по X, компонента ветра X меняет знак
            x[4] = -x[4]
            
        return x, y