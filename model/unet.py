import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import *

class EfficientUNet(nn.Module):
    def __init__(self, encoder_name="efficientnet-b3", pretrained=True):
        super().__init__()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=1,
            classes=1
        )

        self.global_dtm_max_abs = GLOBAL_DTM_MAX_ABS
        self.dtm_prediction_margin = DTM_PREDICTION_MARGIN

    def forward(self, x):
        logits = self.model(x)

        output_normalized = torch.sigmoid(logits)

        denormalized_output = output_normalized * (self.global_dtm_max_abs + self.dtm_prediction_margin)
        
        return denormalized_output