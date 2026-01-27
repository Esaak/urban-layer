import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import datetime
import logging
import json
from tqdm import tqdm
import torch.nn.functional as F
from dataset import UrbanAirDataset
from model import UNet25D
from vis_utils import plot_training_curves
from metrics_utils import AirQualityMetrics

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg['device']
        
        # Создание директории эксперимента
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(cfg['base_outputs_dir'], f"output_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.logger = self._setup_logger()
        self.logger.info(f"Experiment directory: {self.exp_dir}")

        # Модель, лосс, опт
        self.model = UNet25D(n_channels=6, n_classes=5).to(self.device)
        #self.criterion = nn.MSELoss()
        self.criterion = self._criterion

        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.metrics = AirQualityMetrics()
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_fb': [], 'val_r2': [], 'lr': []}

    def _setup_logger(self):
        logger = logging.getLogger("UrbanAir")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(os.path.join(self.exp_dir, 'train.log'))
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger
    
    def _criterion(self, pred, target, x_input):
    # 1. Основной лосс (MSE на логах)
        base_loss = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        
        # 2. Штраф за "призраков" (концентрация внутри зданий)
        # x_input[:, 0:1] — карта зданий [Batch, 1, 128, 128]
        # pred[:, 0:2] — слои Z1 и Z5 [Batch, 2, 128, 128]
        build_mask = (x_input[:, 0:1, :, :] > 0).float()
        
        # Выделяем предсказания в нижних слоях
        ground_layers = pred[:, 0:2, :, :]
        
        # Умножаем предсказания на маску зданий. 
        # Там, где зданий нет, будет 0. Там, где есть — останется значение предсказания.
        # Мы хотим, чтобы эти значения стремились к нулю.
        ghost_values = ground_layers * build_mask
        ghost_penalty = torch.mean(ghost_values ** 2)
        
        # Итоговый лосс с весовым коэффициентом
        return 1.0 * base_loss + 0.05 * mae + 0.1 * ghost_penalty
    

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for x, y in tqdm(loader, desc="Training", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y, x)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        self.metrics.reset()
        total_loss = 0
        with torch.no_grad():
            for x, y_log in tqdm(loader, desc="Validation", leave=False):
                x, y_log = x.to(self.device), y_log.to(self.device)
                pred_log = self.model(x)
                
                loss = self.criterion(pred_log, y_log, x)
                total_loss += loss.item()
                
                # Метрики в реальных величинах
                pred_real = torch.clamp(torch.expm1(pred_log), min=0)
                target_real = torch.expm1(y_log)
                self.metrics.update(pred_real, target_real)
        
        return total_loss / len(loader), self.metrics.compute()

    def run(self, train_loader, val_loader):
        best_val_loss = float('inf')
        self.logger.info(f"Dataset size (samples): \n Train: {len(train_loader.dataset)} \n Val: {len(val_loader.dataset)}")
        for epoch in range(self.cfg['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_loss, m = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Логирование
            self.logger.info(f"Epoch {epoch+1:02d} | Loss: T={train_loss:.5f} V={val_loss:.5f} | "
                             f"MAE: {m['mae']:.3f} | FB: {m['fb']:.3f} | R2: {m['r2']:.3f} | LR: {current_lr:.2e}")
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(m['mae'])
            self.history['val_fb'].append(m['fb'])
            self.history['val_r2'].append(m['r2'])
            self.history['lr'].append(current_lr)
            
            plot_training_curves(self.history, self.exp_dir, self.cfg['exp_name'])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.exp_dir, f"{self.cfg['exp_name']}_best.pth"))

if __name__ == "__main__":
    CONFIG = {
        'data_dir': '/app/urban-layer-datasets/2026_01_19_500_25d_data/',
        'base_outputs_dir': './outputs/',
        'batch_size': 128,
        'lr': 5e-4,
        'epochs': 50,
        'exp_name': 'unet_25d_v1',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    trainer = Trainer(CONFIG)
    train_ds = UrbanAirDataset(CONFIG['data_dir'], mode='train')
    val_ds = UrbanAirDataset(CONFIG['data_dir'], mode='val')
    
    t_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    v_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    trainer.run(t_loader, v_loader)