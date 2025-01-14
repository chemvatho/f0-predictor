# f0_predictor/utils.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union

def plot_f0_comparison(
    time_stamps: np.ndarray,
    original_f0: np.ndarray,
    predicted_f0: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    plt.figure(figsize=(30, 6))
    plt.plot(time_stamps, original_f0, label='Original F0',
             alpha=0.6, marker="o", linestyle="-")
    plt.plot(time_stamps, predicted_f0, label='Predicted F0 (Viterbi)',
             alpha=0.6, marker="o", linestyle="-")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('F0 Contour Comparison with Viterbi Smoothing')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=1200)
    plt.show()

def evaluate_prediction(
    original_f0: np.ndarray,
    predicted_f0: np.ndarray
) -> Optional[Dict[str, float]]:
    voiced_mask = original_f0 > 0
    if np.sum(voiced_mask) == 0:
        return None

    original_voiced = original_f0[voiced_mask]
    predicted_voiced = predicted_f0[voiced_mask]

    mse = np.mean((original_voiced - predicted_voiced) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(original_voiced - predicted_voiced))

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}