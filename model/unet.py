import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class EfficientUNet(nn.Module):
    def __init__(self, encoder_name="efficientnet-b0", pretrained=True):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained(encoder_name) if pretrained else EfficientNet.from_name(encoder_name)

        # EfficientNet-B0: canali dei blocchi usati per skip connection
        self.enc_channels = {
            "x5": 40,
            "x7": 112,
            "x12": 192,
            "x17": 320
        }

        # Decoder
        self.up3 = nn.ConvTranspose2d(self.enc_channels["x17"], self.enc_channels["x12"], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(self.enc_channels["x12"] + self.enc_channels["x12"], 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(64, self.enc_channels["x7"], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(32, self.enc_channels["x5"], kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(80, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        #self.final = nn.Conv2d(16, 1, kernel_size=1)
        self.final = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid() # Aggiungo sigmoid per restare nell'intervallo [0-1]
        )

    def forward(self, x):
        x_input = x  # shape originale per interpolazione finale
        x = x.repeat(1, 3, 1, 1)  # PAN grigia -> RGB finto

        # Encoder
        x0 = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(x)))
        x1 = self.encoder._blocks[0](x0)
        x2 = self.encoder._blocks[1](x1)
        x3 = self.encoder._blocks[2](x2)
        x4 = self.encoder._blocks[3](x3)
        x5 = self.encoder._blocks[4](x4)  # skip 1
        x6 = self.encoder._blocks[5](x5)
        x7 = self.encoder._blocks[6](x6)  # skip 2
        x8 = self.encoder._blocks[7](x7)
        x9 = self.encoder._blocks[8](x8)
        x10 = self.encoder._blocks[9](x9)
        x11 = self.encoder._blocks[10](x10)
        x12 = self.encoder._blocks[11](x11)  # skip 3
        x13 = self.encoder._blocks[12](x12)
        x14 = self.encoder._blocks[13](x13)
        x15 = self.encoder._blocks[14](x14)
        x16 = self.encoder._blocks[15](x15)
        x17 = x16  # bottleneck

        # Decoder
        x = self.up3(x17)
        x = F.interpolate(x, size=x12.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, x12], dim=1))

        x = self.up2(x)
        x = F.interpolate(x, size=x7.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, x7], dim=1))

        x = self.up1(x)
        x = F.interpolate(x, size=x5.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, x5], dim=1))

        x = self.final(x)
        x = F.interpolate(x, size=x_input.shape[-2:], mode="bilinear", align_corners=False)

        return x
