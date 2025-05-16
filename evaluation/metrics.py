import numpy as np
import torch

def rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def nmad(pred, gt):
    return 1.4826 * np.median(np.abs(pred - gt - np.median(pred - gt)))

def delta_metrics(pred, target, thresholds=[1.25, 1.25**2, 1.25**3]):
    """
    Calcola le metriche delta1, delta2, delta3 tra predizione e target.
    
    Args:
        pred (torch.Tensor): Tensor di predizioni (B x 1 x H x W)
        target (torch.Tensor): Tensor ground truth (B x 1 x H x W)
        thresholds (list): soglie per le metriche delta
    
    Returns:
        dict: {'delta1': ..., 'delta2': ..., 'delta3': ...}
    """
    pred = pred.clamp(min=1e-8)
    target = target.clamp(min=1e-8)

    ratio = torch.max(pred / target, target / pred)
    
    metrics = {}
    for i, t in enumerate(thresholds):
        metrics[f'delta{i+1}'] = (ratio < t).float().mean().item()

    return metrics