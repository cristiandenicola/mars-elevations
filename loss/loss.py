import torch
import torch.nn.functional as F

def edge_loss(pred, target):
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=pred.dtype, device=pred.device).unsqueeze(0).unsqueeze(0)

    def apply_sobel(x):
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # evita sqrt(0)

    pred_edge = apply_sobel(pred)
    target_edge = apply_sobel(target)

    return F.l1_loss(pred_edge, target_edge)

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

def combined_loss(alpha=0.5, beta=0.3, gamma=0.2):
    def loss_fn(pred, target):
        charbonnier = charbonnier_loss(pred, target)
        grad = gradient_difference_loss(pred, target)
        edge = edge_loss(pred, target)
        return alpha * charbonnier + beta * grad + gamma * edge
    return loss_fn

