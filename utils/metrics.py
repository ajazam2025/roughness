import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MSE": mse
    }
