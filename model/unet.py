import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

class EfficientUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNet.from_pretrained("efficientnet-b0") # modello usato: EfficientUNet

        # layer di convoluzione trasposta 2D
        # utilizzata per aumentare la risoluzione spaziale delle features.
        # params: 320 input chan, 112 output chan, 2x2 kernel, upsampling 2x
        self.up3 = nn.ConvTranspose2d(320, 112, kernel_size=2, stride=2) 

        # blocco sequenziale 
        # conv2D con 112 + 192 input chan concatenazione con out up3 (112 c) e layer encoder (192 c) - out 64 chan, 3x3 kernel
        # activation func relu applicata in-place (salvo memoria applicando direttamente su input)
        # altra convoluzione 2D con 64 chan in input e output, kernel 3x3
        # altra relu in-place
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

        # ultimo layer convoluzionale
        # 1 out > regress mappa di elevazione (DTM)
        self.final = nn.Conv2d(16, 1, 1)

    # definisce come i dati x (in) vengono processati attraverso la rete
    def forward(self, x):
        x_input = x  # salviamo per l'upsample finale

        # input x ha un solo canale (scala di grigi), questa linea lo ripete lungo la dim dei chan per creare tensore con 3 chan
        # necessario perché encoder EfficientNet pre-addestrato aspetta input a 3 canali (RGB)
        x = x.repeat(1, 3, 1, 1)

        # propago x attraverso i primi layer e alcuni dei blocchi dell'encoder EfficientNet
        # salvo output in alcuni blocchi intermedi (x5, x7, x12, x17)
        # questi output verranno utilizzati per le skip connection nel decoder U-Net
        # # enc1, # enc2, # enc3 sono i livelli dell'encoder da cui vengono estratte le features
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

        # out x17 dell'encoder viene sottoposto a convoluzione trasposta (self.up3) per aumentare la sua res spaziale
        # poi interpolazione bilineare (F.interpolate) per adattare ulteriormente le dimensioni spaziali di x 
        # alle dimensioni spaziali dell'output del blocco x12 dell'encoder. 

        # out upsampled x viene concatenato con la mappa delle caratteristiche x12 lungo la dimensione dei canali (dim=1)
        # permette al decoder di accedere alle caratteristiche a bassa risoluzione ma con contesto dell'encoder
        # tensor concatenato viene passato attraverso il blocco di decodifica self.dec3 (2 conv 3x3 con ReLU).
        x = self.up3(x17)
        x = F.interpolate(x, size=x12.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, x12], dim=1))

        # uso out di dec3, upsampling con up2
        # interpolando alla dimensione di x7, concatenando con x7 e passando attraverso dec2
        x = self.up2(x)
        x = F.interpolate(x, size=x7.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, x7], dim=1))

        # stessa cosa con out dec2, upsampling con up1
        # interpolate a x5, concatenando a x5
        x = self.up1(x)
        x = F.interpolate(x, size=x5.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, x5], dim=1))

        # output
        # out di dec1 viene passato attraverso la convoluzione finale self.final per produrre DTM a single chan
        # out viene interpolato alle dimensioni spaziali dell'input originale x_input
        out = self.final(x)
        out = F.interpolate(out, size=x_input.shape[-2:], mode='bilinear', align_corners=False)
        return out
