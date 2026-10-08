import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

def evaluate_model(y_test, predictions):
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse)
    }
