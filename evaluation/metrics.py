import numpy as np
import torch
import torch.nn.functional as F

def rmse(pred: torch.Tensor, gt: torch.Tensor):
    return torch.sqrt(F.mse_loss(pred, gt, reduction="mean"))

def mae(pred: torch.Tensor, gt: torch.Tensor):
    return F.l1_loss(pred, gt, reduction="mean")

def nmad(pred: torch.Tensor, gt: torch.Tensor):
    diff = pred - gt
    med = diff.median()
    mad = (diff - med).abs().median()
    return 1.4826 * mad

def delta_metrics_torch(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    thresholds=(1.25, 1.25**2, 1.25**3), 
    epsilon=1e-6
):
    # Evita divisione per zero
    target_safe = target.clamp(min=epsilon)
    pred_safe   = pred.clamp(min=epsilon)

    ratio = torch.max(pred_safe / target_safe, target_safe / pred_safe)
    metrics = {}
    for i, t in enumerate(thresholds, 1):
        metrics[f"delta{i}"] = (ratio < t).float().mean()
    return metrics

def compute_metrics(pred: torch.Tensor, gt: torch.Tensor):
    metrics = {
        "rmse": rmse(pred, gt).item(),
        "mae": mae(pred, gt).item(),
        "nmad": nmad(pred, gt).item(),
    }
    metrics.update({k: v.item() for k, v in delta_metrics_torch(pred, gt).items()})
    return metrics