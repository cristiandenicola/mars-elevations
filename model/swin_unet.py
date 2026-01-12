import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from config import *

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class EnhancedFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels * 2, channels, 1)
        
        self.conv_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        
        # Attention modules
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        
        # Residual connection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_deep, skip_feature):
        if x_deep.shape[-2:] != skip_feature.shape[-2:]:
            x_deep = F.interpolate(x_deep, size=skip_feature.shape[-2:], 
                                 mode='bilinear', align_corners=True)

        x = torch.cat([x_deep, skip_feature], dim=1)
        x = self.conv_reduce(x)
        
        # Residual connection
        identity = x
        
        # Feature refinement
        x = self.conv_refine(x)
        
        # Add residual
        x = x + identity
        x = self.relu(x)
        
        # Apply attention
        x = self.channel_att(x)
        x = self.spatial_att(x)
        
        return x

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.project(x)
        
        return x

class EnhancedSwinDepth(nn.Module):
    def __init__(self, pretrained=True, global_max_depth=GLOBAL_DTM_MAX_ABS, input_size=256):
        super().__init__()
        self.global_dtm_max_abs = global_max_depth
        self.input_size = input_size
        DECODER_CHANNELS = 256

        # Encoder
        self.encoder = timm.create_model(
            'swinv2_base_window16_256.ms_in1k',
            pretrained=pretrained,
            features_only=True,
        )

        # Infer encoder channels
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, input_size, input_size)
            feats = self.encoder(dummy)
        encoder_channels = [f.shape[1] for f in feats]

        # ASPP
        self.aspp = ASPP(encoder_channels[-1], DECODER_CHANNELS)

        # Projection blocks
        self.projection_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, DECODER_CHANNELS, 1),
                nn.BatchNorm2d(DECODER_CHANNELS),
                nn.ReLU(inplace=True)
            ) for in_ch in encoder_channels[:-1]
        ])

        # Enhanced fusion blocks
        self.fusion_blocks = nn.ModuleList([
            EnhancedFusionBlock(DECODER_CHANNELS) for _ in range(len(encoder_channels) - 1)
        ])

        # Multi-scale prediction heads (supervision)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(DECODER_CHANNELS, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1)
            ) for _ in range(len(encoder_channels) - 1)
        ])

        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Conv2d(DECODER_CHANNELS, DECODER_CHANNELS // 2, 3, padding=1),
            nn.BatchNorm2d(DECODER_CHANNELS // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(DECODER_CHANNELS // 2, DECODER_CHANNELS // 4, 3, padding=1),
            nn.BatchNorm2d(DECODER_CHANNELS // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(DECODER_CHANNELS // 4, 1, 1),
        )

    def forward(self, x):
        original_size = x.shape[-2:]
        
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Encoder
        features = self.encoder(x)
        
        # Process bottleneck con ASPP
        x_decoder = self.aspp(features[-1])
        
        # Project skip connections
        projected = []
        for proj, f in zip(self.projection_blocks, features[:-1]):
            projected.append(proj(f))

        skip_feats = list(reversed(projected))
        aux_outputs = []
        
        for i, (fusion_block, aux_head, skip_feat) in enumerate(
            zip(self.fusion_blocks, self.aux_heads, skip_feats)
        ):
            x_decoder = fusion_block(x_decoder, skip_feat)
            
            # Auxiliary output per multi-scale supervision
            if self.training:
                aux_out = aux_head(x_decoder)
                aux_out = F.interpolate(aux_out, size=original_size, mode='bilinear', align_corners=True)
                aux_outputs.append(torch.sigmoid(aux_out) * self.global_dtm_max_abs)

        # Final prediction
        logits = self.prediction_head(x_decoder)
        
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=True)

        main_output = torch.sigmoid(logits) * self.global_dtm_max_abs
        
        if self.training and aux_outputs:
            return main_output, aux_outputs
        else:
            return main_output