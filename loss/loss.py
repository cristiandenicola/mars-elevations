import torch
import torch.nn.functional as F

def gradient_difference_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_dx = torch.abs(pred_dx - target_dx)
    loss_dy = torch.abs(pred_dy - target_dy)
    return torch.mean(loss_dx) + torch.mean(loss_dy)

def charbonnier_loss(pred, target, epsilon=1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff ** 2 + epsilon ** 2))

def combined_loss(pred, target, alpha=0.9):
    data_loss = charbonnier_loss(pred, target)
    grad_loss = gradient_difference_loss(pred, target)
    return alpha * data_loss + (1 - alpha) * grad_loss

