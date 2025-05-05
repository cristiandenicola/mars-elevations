import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(pred, gt):
    # scarto NaN
    pred = np.nan_to_num(pred, nan=0.0)
    gt = np.nan_to_num(gt, nan=0.0)
    
    # Calcola le metriche
    try:
        rmse = np.sqrt(mean_squared_error(gt, pred))
        mae = mean_absolute_error(gt, pred)
        
        # Calcola anche la correlazione se possibile
        correlation = np.corrcoef(gt.flatten(), pred.flatten())[0, 1]
        
        print(f"Evaluation Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Correlation: {correlation:.4f}")
        
        return {
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation
        }
    except Exception as e:
        print(f"Errore durante la valutazione: {e}")
        print(f"Info debug:")
        print(f"  Pred shape: {pred.shape}, GT shape: {gt.shape}")
        print(f"  Pred contiene NaN: {np.isnan(pred).any()}")
        print(f"  GT contiene NaN: {np.isnan(gt).any()}")
        print(f"  Pred min/max: {np.min(pred)}/{np.max(pred)}")
        print(f"  GT min/max: {np.min(gt)}/{np.max(gt)}")
        return None