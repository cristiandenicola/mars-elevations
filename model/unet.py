import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 
from config import *

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        
        if skip_channels > 0:
            self.skip_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
            conv_in_channels = out_channels * 2 
        else:
            self.skip_conv = None
            conv_in_channels = out_channels
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None and self.skip_conv is not None:
            skip = self.skip_conv(skip)
            
            if x.shape[-2:] != skip.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv_block(x)
        return x


class SwinUNetRegressor(nn.Module):
    def __init__(self, encoder_name=ENCODER_NAME, pretrained=PRETRAINED):
        super().__init__()
        
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,  
            in_chans=1          
        )
        
        encoder_channels = self.encoder.feature_info.channels()
        
        self.decoder = self._build_decoder(encoder_channels)
        
        self.target_dtm_normalization_range = TARGET_DTM_NORMALIZATION_RANGE
    
    def _build_decoder(self, encoder_channels):
        decoder_channels = [512, 256, 128, 64] 

        center_in_channels = encoder_channels[-1] 
        center_out_channels = decoder_channels[0] 
        self.center = nn.Sequential(
            nn.Conv2d(center_in_channels, center_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(center_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(center_out_channels, center_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(center_out_channels),
            nn.ReLU(inplace=True)
        )
        
        decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_channels = center_out_channels if i == 0 else decoder_channels[i-1]
            skip_idx = len(encoder_channels) - (i + 2) 
            skip_channels = encoder_channels[skip_idx] if skip_idx >= 0 else 0
            out_channels = decoder_channels[i] 
            
            decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels
                )
            )
        
        self.final_conv = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)
        
        return nn.ModuleDict({
            'center': self.center, 
            'blocks': decoder_blocks,
            'final': self.final_conv
        })
    
    def forward(self, x):
        input_size = x.shape[-2:]  
        
        features = self.encoder(x)

        # --- PERMUTA DA NHWC A NCHW ---
        features_nchw = [f.permute(0, 3, 1, 2).contiguous() for f in features]
        
        x = self.decoder['center'](features_nchw[-1]) 
        
        for i, decoder_block in enumerate(self.decoder['blocks']):
            skip_idx = len(features_nchw) - 2 - i 
            skip = features_nchw[skip_idx] if skip_idx >= 0 else None
            x = decoder_block(x, skip)
        
        logits = self.decoder['final'](x)
        
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        output_normalized_0_1 = torch.sigmoid(logits)
        output_in_target_range = output_normalized_0_1 * self.target_dtm_normalization_range
        
        return output_in_target_range