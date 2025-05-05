import numpy as np

def rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def nmad(pred, gt):
    return 1.4826 * np.median(np.abs(pred - gt - np.median(pred - gt)))