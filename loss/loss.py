import torch
import torch.nn.functional as F

def sobel_filters(device):
    sobel_x = torch.tensor([[[[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]]]], dtype=torch.float32).to(device)

    sobel_y = torch.tensor([[[[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]]]], dtype=torch.float32).to(device)

    return sobel_x, sobel_y

def edge_loss(pred, target):
    device = pred.device
    sobel_x, sobel_y = sobel_filters(device)

    pred_x = F.conv2d(pred, sobel_x, padding=1)
    pred_y = F.conv2d(pred, sobel_y, padding=1)
    pred_edges = torch.sqrt(pred_x ** 2 + pred_y ** 2)

    target_x = F.conv2d(target, sobel_x, padding=1)
    target_y = F.conv2d(target, sobel_y, padding=1)
    target_edges = torch.sqrt(target_x ** 2 + target_y ** 2)

    return F.l1_loss(pred_edges, target_edges)

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

def combined_loss(alpha=0.6, beta=0.25, gamma=0.15):
    def loss_fn(pred, target):
        charbonnier = charbonnier_loss(pred, target)
        grad = gradient_difference_loss(pred, target)
        edge = edge_loss(pred, target)
        return alpha * charbonnier + beta * grad + gamma * edge
    return loss_fn

