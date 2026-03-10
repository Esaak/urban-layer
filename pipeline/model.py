import torch
import torch.nn as nn
import torch.nn.functional as F

class ModulatedDoubleConv(nn.Module):
    """Блок, который умеет принимать параметры ветра для модуляции признаков"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.Mish(inplace=True)

    def forward(self, x, gamma, beta):
        # Первый слой свертки
        x = self.conv1(x)
        x = self.norm1(x)
        # Модуляция ветром: x = x * (1 + gamma) + beta
        
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        
        x = x * (1 + gamma) + beta
        x = self.act(x)
        
        # Второй слой свертки
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x

class UNet25D(nn.Module):
    def __init__(self, n_channels=8, n_classes=5, wind_dim=3):
        super().__init__()
        
        # 1. Глобальный эмбеддинг ветра
        # Он создает "смысловой вектор" ветра, из которого мы нарежем параметры для каждого слоя
        self.wind_encoder = nn.Sequential(
            nn.Linear(wind_dim, 128),
            nn.Mish(),
            nn.Linear(128, 512) # Общий банк признаков ветра
        )

        # Проекторы из эмбеддинга ветра в конкретные слои декодера (gamma и beta)
        # Нам нужно 2 * число_каналов для каждого уровня декодера
        self.proj3 = nn.Linear(512, 256 * 2) # Для слоя 256
        self.proj2 = nn.Linear(512, 128 * 2) # Для слоя 128
        self.proj1 = nn.Linear(512, 64 * 2)  # Для слоя 64

        # --- Encoder (Геометрия) ---
        self.inc = nn.Sequential(nn.Conv2d(n_channels + 4, 64, 3, padding=1), nn.Mish())
        self.down1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.Mish())
        self.down2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.Mish())
        self.down3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(256, 512, 3, padding=1), nn.Mish())
        
        # --- Decoder (С модуляцией на каждом шаге) ---
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = ModulatedDoubleConv(512, 256) # skip(256) + up(256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ModulatedDoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ModulatedDoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def add_coords(self, x):
        # CoordConv жизненно важен для ветра! Без него сеть не понимает направления
        batch_size, _, h, w = x.size()
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing='ij')
        grid_y = grid_y.view(1, 1, h, w).expand(batch_size, -1, -1, -1).to(x.device)
        grid_x = grid_x.view(1, 1, h, w).expand(batch_size, -1, -1, -1).to(x.device)
        return torch.cat([x, grid_x, grid_y, 1-grid_x, 1-grid_y], dim=1)

    def forward(self, x, wind):
        # 0. Координаты и Эмбеддинг ветра
        x = self.add_coords(x)
        w_emb = self.wind_encoder(wind)
        
        # 1. Encoder (Skip-connections нужны!)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 2. Decoder с послойным внедрением ветра
        
        # Уровень 1 (256 каналов)
        d1 = self.up1(x4)
        d1 = torch.cat([x3, d1], dim=1)
        p1 = self.proj3(w_emb)
        gamma1, beta1 = torch.chunk(p1, 2, dim=1)
        d1 = self.dec1(d1, gamma1, beta1)
        
        # Уровень 2 (128 каналов)
        d2 = self.up2(d1)
        d2 = torch.cat([x2, d2], dim=1)
        p2 = self.proj2(w_emb)
        gamma2, beta2 = torch.chunk(p2, 2, dim=1)
        d2 = self.dec2(d2, gamma2, beta2)
        
        # Уровень 3 (64 канала)
        d3 = self.up3(d2)
        d3 = torch.cat([x1, d3], dim=1)
        p3 = self.proj1(w_emb)
        gamma3, beta3 = torch.chunk(p3, 2, dim=1)
        d3 = self.dec3(d3, gamma3, beta3)
        
        return self.outc(d3)
# class DoubleConv(nn.Module):
#     """(Conv -> BN -> ReLU) * 2"""
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             # nn.BatchNorm2d(out_ch),
#             nn.Mish(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             # nn.BatchNorm2d(out_ch),
#             nn.Mish(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class FiLM(nn.Module):
#    """Модуляция признаков: x = x * gamma + beta"""
#    def forward(self, x, gamma, beta):
#        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
#        beta = beta.view(x.size(0), x.size(1), 1, 1)
#        return x * gamma + beta

# class UNet25D(nn.Module):
#     def __init__(self, n_channels=8, n_classes=5, wind_dim=3):
#         super().__init__()
#         self.in_channels = n_channels
        
#         # --- Encoder ---
#         self.inc = DoubleConv(self.in_channels, 64)
#         self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
#         self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
#         self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
#         # --- Bottleneck + Wind FFN ---
#         self.bottleneck = DoubleConv(512, 512)
        
#         # FFN для обработки глобального вектора ветра (cos, sin, mag)
#         self.wind_ffn = nn.Sequential(
#             nn.Linear(wind_dim, 128),
#             nn.Mish(),
#             nn.Linear(128, 512*2) # Генерируем Gamma и Beta для 512 каналов
#         )
        
        
#         self.film = FiLM()
        
#         # --- Decoder ---
#         self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv1 = DoubleConv(512, 256) # 256 (skip) + 256 (up)
        
#         self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv2 = DoubleConv(256, 128)
        
#         self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv3 = DoubleConv(128, 64)
        
#         self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        

#     def forward(self, x, wind):
        
#         # 2. Encoder
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
        
#         # 3. Bottleneck + FiLM Modulation
#         x_bottleneck = self.bottleneck(x4)
        
#         # Генерируем параметры модуляции из ветра
#         wind_params = self.wind_ffn(wind)
#         gamma, beta = torch.chunk(wind_params, 2, dim=1)
        
#         # Модулируем признаки в горлышке
#         x_modulated = self.film(x_bottleneck, gamma, beta)
#         # 4. Decoder
        
#         # d1 = self.up1(x_bottleneck + wind_params.view(x_bottleneck.size(0), x_bottleneck.size(1), 1, 1))
#         d1 = self.up1(x_modulated)
#         d1 = torch.cat([x3, d1], dim=1)
#         d1 = self.conv1(d1)
        
#         d2 = self.up2(d1)
#         d2 = torch.cat([x2, d2], dim=1)
#         d2 = self.conv2(d2)
        
#         d3 = self.up3(d2)
#         d3 = torch.cat([x1, d3], dim=1)
#         d3 = self.conv3(d3)
        
#         return self.outc(d3)
    

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DoubleConv(nn.Module):
#     """(Conv -> BN -> ReLU) * 2"""
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             # nn.BatchNorm2d(out_ch),
#             nn.Mish(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             # nn.BatchNorm2d(out_ch),
#             nn.Mish(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class UNet25D(nn.Module):
#     # def __init__(self, n_channels=6, n_classes=5):
#     def __init__(self, n_channels=7, n_classes=5):
#         super().__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         # --- Encoder ---
#         # 64 -> 128 -> 256 -> 512 -> 1024 (Bottleneck)
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
#         self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
#         self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
#         self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
#         # --- Decoder ---
#         self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.conv1 = DoubleConv(1024, 512) # 512 from up + 512 from skip
        
#         self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.conv2 = DoubleConv(512, 256)
        
#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.conv3 = DoubleConv(256, 128)
        
#         self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.conv4 = DoubleConv(128, 64)
        
#         # --- Output Head ---
#         # Выдает 5 каналов: [Z1, Z5, Z8, Z25, Mean]
#         self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
#     def forward(self, x):
#         # Encoder
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
        
#         # Decoder
#         x = self.up1(x5)
#         x = torch.cat([x4, x], dim=1) # Skip connection
#         x = self.conv1(x)
        
#         x = self.up2(x)
#         x = torch.cat([x3, x], dim=1)
#         x = self.conv2(x)
        
#         x = self.up3(x)
#         x = torch.cat([x2, x], dim=1)
#         x = self.conv3(x)
        
#         x = self.up4(x)
#         x = torch.cat([x1, x], dim=1)
#         x = self.conv4(x)
        
#         logits = self.outc(x)
#         return logits



# class ResidualBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, groups=8):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.GroupNorm(groups, out_ch),
#             nn.GELU(),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.GroupNorm(groups, out_ch)
#         )
#         self.shortcut = nn.Sequential()
#         if in_ch != out_ch:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, 1, bias=False),
#                 nn.GroupNorm(groups, out_ch)
#             )
#         self.final_act = nn.GELU()

#     def forward(self, x):
#         return self.final_act(self.conv(x) + self.shortcut(x))

