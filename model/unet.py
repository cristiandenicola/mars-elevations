import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

class EfficientUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained("efficientnet-b0")

        self.up3 = nn.ConvTranspose2d(320, 112, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(112 + 192, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(64, 40, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(120, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x_input = x  # salviamo per l'upsample finale
        x = x.repeat(1, 3, 1, 1)

        x0 = self.encoder._conv_stem(x)
        x0 = self.encoder._bn0(x0)
        x0 = self.encoder._swish(x0)

        x1 = self.encoder._blocks[0](x0)
        x2 = self.encoder._blocks[1](x1)
        x3 = self.encoder._blocks[2](x2)
        x4 = self.encoder._blocks[3](x3)
        x5 = self.encoder._blocks[4](x4)  # enc1
        x6 = self.encoder._blocks[5](x5)
        x7 = self.encoder._blocks[6](x6)  # enc2
        x8 = self.encoder._blocks[7](x7)
        x9 = self.encoder._blocks[8](x8)
        x10 = self.encoder._blocks[9](x9)
        x11 = self.encoder._blocks[10](x10)
        x12 = self.encoder._blocks[11](x11)  # enc3
        x13 = self.encoder._blocks[12](x12)
        x14 = self.encoder._blocks[13](x13)
        x15 = self.encoder._blocks[14](x14)
        x16 = self.encoder._blocks[15](x15)
        x17 = x16

        x = self.up3(x17)
        x = F.interpolate(x, size=x12.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, x12], dim=1))

        x = self.up2(x)
        x = F.interpolate(x, size=x7.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, x7], dim=1))

        x = self.up1(x)
        x = F.interpolate(x, size=x5.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, x5], dim=1))

        out = self.final(x)
        out = F.interpolate(out, size=x_input.shape[-2:], mode='bilinear', align_corners=False)
        return out
