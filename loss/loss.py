import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import *

def edge_loss(pred, target, mask):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                           dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                           dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)

    def apply_sobel(x):
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    pred_edge = apply_sobel(pred)
    target_edge = apply_sobel(target)

    masked_loss = F.l1_loss(pred_edge, target_edge, reduction='none') * mask
    return masked_loss.sum() / (mask.sum() + 1e-8)

def gradient_difference_loss(pred, target, mask):
    mask_dx = mask[:, :, :, 1:]
    mask_dy = mask[:, :, 1:, :]
    
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    # Calcola la loss dei gradienti solo sulle aree mascherate
    loss_dx = (torch.abs(pred_dx - target_dx) * mask_dx).sum()
    loss_dy = (torch.abs(pred_dy - target_dy) * mask_dy).sum()
    
    total_mask = mask_dx.sum() + mask_dy.sum() + 1e-8
    return (loss_dx + loss_dy) / total_mask

def charbonnier_loss(pred, target, mask, epsilon=1e-3):
    diff = (pred - target) * mask
    masked_loss = torch.sqrt(diff ** 2 + epsilon ** 2)
    return masked_loss.sum() / (mask.sum() + 1e-8)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGGPerceptualLoss, self).__init__()
        vgg19 = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features
        self.slice1 = nn.Sequential()
        for x in range(21):
            self.slice1.add_module(str(x), vgg19[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False

        #self.register_buffer('depth_mean', torch.tensor([50.0]).view(1, 1, 1, 1))
        #self.register_buffer('depth_std', torch.tensor([50.0]).view(1, 1, 1, 1))

        self.register_buffer('depth_mean', torch.tensor([GLOBAL_DTM_MEAN]).view(1, 1, 1, 1))
        self.register_buffer('depth_std', torch.tensor([GLOBAL_DTM_STD]).view(1, 1, 1, 1))
        
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, pred, target):
        pred_normalized = (pred - self.depth_mean) / self.depth_std
        target_normalized = (target - self.depth_mean) / self.depth_std
        
        pred_rgb = pred_normalized.repeat(1, 3, 1, 1)
        target_rgb = target_normalized.repeat(1, 3, 1, 1)

        pred_vgg_normalized = (pred_rgb - self.vgg_mean) / self.vgg_std
        target_vgg_normalized = (target_rgb - self.vgg_mean) / self.vgg_std
        
        pred_features = self.slice1(pred_vgg_normalized)
        target_features = self.slice1(target_vgg_normalized)
        return F.l1_loss(pred_features, target_features)

def combined_loss_with_perceptual(alpha=0.4, beta=0.4, gamma=0.2, delta=1.0, mae_weight=0.5, perceptual_loss_fn=None):
    def loss_fn(pred, target):
        mask = (target != 0).float()
        
        mae = F.l1_loss(pred, target, reduction='none') * mask
        mae_loss = mae.sum() / (mask.sum() + 1e-8)
        
        charbonnier = charbonnier_loss(pred, target, mask)
        grad = gradient_difference_loss(pred, target, mask)
        edge = edge_loss(pred, target, mask)
        
        perceptual = 0.0
        if perceptual_loss_fn:
            perceptual = perceptual_loss_fn(pred, target)
        
        return (mae_loss * mae_weight +
                alpha * charbonnier +
                beta * grad +
                gamma * edge +
                delta * perceptual)
    return loss_fn