import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from config import *

class EfficientUNet(nn.Module):
    def __init__(self, encoder_name="efficientnet-b0", pretrained=True):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained(encoder_name) if pretrained else EfficientNet.from_name(encoder_name)

        # EfficientNet-B0: canali dei blocchi usati per skip connection
        self.enc_channels = {
            "x5": 40,
            "x7": 80,
            "x12": 192,
            "x17": 320 # Bottleneck
        }

        # up3: Upsampling e primo Conv2d per ridurre i canali da bottleneck a x12
        self.up_block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # Upsampling
            nn.Conv2d(self.enc_channels["x17"], self.enc_channels["x12"], kernel_size=3, padding=1), # 320 -> 192 canali
            nn.BatchNorm2d(self.enc_channels["x12"]),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            # Input channels: canali upsamplati (x12) + canali skip connection (x12) = 192 + 192 = 384
            nn.Conv2d(self.enc_channels["x12"] + self.enc_channels["x12"], 128, kernel_size=3, padding=1), # 384 -> 128 canali
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # up2: Upsampling e primo Conv2d per ridurre i canali da dec3_output a x7
        self.up_block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, self.enc_channels["x7"], kernel_size=3, padding=1), # 64 -> 80 canali 
            nn.BatchNorm2d(self.enc_channels["x7"]),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            # Input channels: canali upsamplati (x7) + canali skip connection (x7) = 112 + 112 = 224
            nn.Conv2d(self.enc_channels["x7"] + self.enc_channels["x7"], 64, kernel_size=3, padding=1), # 160 -> 64 canali
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # up1: Upsampling e primo Conv2d per ridurre i canali da dec2_output a x5
        self.up_block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, self.enc_channels["x5"], kernel_size=3, padding=1), # 64 -> 40 canali
            nn.BatchNorm2d(self.enc_channels["x5"]),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            # Input channels: canali upsamplati (x5) + canali skip connection (x5) = 40 + 40 = 80
            nn.Conv2d(self.enc_channels["x5"] + self.enc_channels["x5"], 32, kernel_size=3, padding=1), # 80 -> 32 canali
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.global_dtm_max_abs = GLOBAL_DTM_MAX_ABS
        self.dtm_prediction_margin = DTM_PREDICTION_MARGIN

    def forward(self, x):
        x_input = x
        x = x.repeat(1, 3, 1, 1)

        # Encoder (senza modifiche)
        x0 = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(x)))
        x1 = self.encoder._blocks[0](x0)
        x2 = self.encoder._blocks[1](x1)
        x3 = self.encoder._blocks[2](x2)
        x4 = self.encoder._blocks[3](x3)
        x5 = self.encoder._blocks[4](x4)  # skip 1 (40 canali)
        x6 = self.encoder._blocks[5](x5)
        x7 = self.encoder._blocks[6](x6)  # skip 2 (80 canali)
        x8 = self.encoder._blocks[7](x7)
        x9 = self.encoder._blocks[8](x8)
        x10 = self.encoder._blocks[9](x9)
        x11 = self.encoder._blocks[10](x10)
        x12 = self.encoder._blocks[11](x11)  # skip 3 (192 canali)
        x13 = self.encoder._blocks[12](x12)
        x14 = self.encoder._blocks[13](x13)
        x15 = self.encoder._blocks[14](x14)
        x16 = self.encoder._blocks[15](x15)
        x17 = x16  # bottleneck (320 canali)

        # up_block3 -> dec3 (da x17 a risoluzione x12)
        x = self.up_block3(x17)
        if x.shape[-2:] != x12.shape[-2:]:
            x = F.interpolate(x, size=x12.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, x12], dim=1)) # Concatena le feature upsampliate con la skip connection

        # up_block2 -> dec2 (da dec3_output a risoluzione x7)
        x = self.up_block2(x)
        if x.shape[-2:] != x7.shape[-2:]:
            x = F.interpolate(x, size=x7.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, x7], dim=1))

        # up_block1 -> dec1 (da dec2_output a risoluzione x5)
        x = self.up_block1(x)
        if x.shape[-2:] != x5.shape[-2:]:
            x = F.interpolate(x, size=x5.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, x5], dim=1))

        x = self.final(x)
        # Interpolazione finale alla dimensione dell'input originale se le dimensioni non sono uguali
        if x.shape[-2:] != x_input.shape[-2:]:
            x = F.interpolate(x, size=x_input.shape[-2:], mode="bilinear", align_corners=False)

        x = x * (self.global_dtm_max_abs + self.dtm_prediction_margin)
        
        return x