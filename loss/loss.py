import torch
import torch.nn.functional as F

def gradient_loss(pred):
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    return torch.mean(dx) + torch.mean(dy)

def combined_loss(pred, target, alpha=0.9):
    mse = F.mse_loss(pred, target) # prossimo test: provare huber loss
    grad = gradient_loss(pred)
    return alpha * mse + (1 - alpha) * grad

