import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def edge_loss(pred, target, mask):
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)

    def apply_sobel(x):
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    pred_edge = apply_sobel(pred)
    target_edge = apply_sobel(target)

    masked_loss = F.l1_loss(pred_edge * mask, target_edge * mask, reduction='sum')
    return masked_loss / (torch.sum(mask) + 1e-6)


def gradient_difference_loss(pred, target, mask):
    pred_dx = (pred * mask)[:, :, :, 1:] - (pred * mask)[:, :, :, :-1]
    pred_dy = (pred * mask)[:, :, 1:, :] - (pred * mask)[:, :, :-1, :]

    target_dx = (target * mask)[:, :, :, 1:] - (target * mask)[:, :, :, :-1]
    target_dy = (target * mask)[:, :, 1:, :] - (target * mask)[:, :, :-1, :]
    
    # La maschera deve essere anche ridotta per i gradienti
    mask_dx = mask[:, :, :, 1:]
    mask_dy = mask[:, :, 1:, :]

    loss_dx = torch.sum(torch.abs(pred_dx - target_dx) * mask_dx)
    loss_dy = torch.sum(torch.abs(pred_dy - target_dy) * mask_dy)

    return (loss_dx + loss_dy) / (torch.sum(mask_dx) + torch.sum(mask_dy) + 1e-6)


def charbonnier_loss(pred, target, mask, epsilon=1e-3):
    diff = (pred - target) * mask
    masked_loss = torch.sum(torch.sqrt(diff ** 2 + epsilon ** 2))
    return masked_loss / (torch.sum(mask) + 1e-6)

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

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Il tuo input ha 1 canale, VGG ne vuole 3. Lo duplichiamo.
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)

        pred_normalized = (pred_rgb - self.mean) / self.std
        target_normalized = (target_rgb - self.mean) / self.std

        pred_features = self.slice1(pred_normalized)
        target_features = self.slice1(target_normalized)
        
        return torch.mean(torch.abs(pred_features - target_features))


def combined_loss_with_perceptual(alpha=0.4, beta=0.4, gamma=0.2, delta=0.001, perceptual_loss_fn=None):
    def loss_fn(pred, target):
        mask = (target != 0).float()
        
        charbonnier = charbonnier_loss(pred, target, mask)
        grad = gradient_difference_loss(pred, target, mask)
        edge = edge_loss(pred, target, mask)
        
        perceptual = 0.0
        if perceptual_loss_fn:
            perceptual = perceptual_loss_fn(pred, target)

        return (alpha * charbonnier +
                beta * grad +
                gamma * edge +
                delta * perceptual)
    return loss_fn