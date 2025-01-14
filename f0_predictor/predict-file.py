# f0_predictor/predict.py
import torch
import numpy as np
import parselmouth
from typing import Tuple, Optional
from .model import F0PredictionModel
from .smoother import ViterbiF0Smoother

def predict_f0_for_audio(
    audio_path: str,
    model: F0PredictionModel,
    sequence_length: int = 50,
    batch_size: int = 128,
    time_step: float = 0.01,
    pitch_floor: float = 75,
    pitch_ceiling: float = 600
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    time_stamps, original_f0 = extract_f0_for_prediction(
        audio_path, time_step, pitch_floor, pitch_ceiling
    )
    predictions = np.zeros_like(original_f0)
    
    chunk_size = 1000
    for chunk_start in range(0, len(original_f0), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(original_f0))
        chunk_length = chunk_end - chunk_start
        
        sequences = create_sequences(
            original_f0, chunk_start, chunk_length, sequence_length
        )
        predictions[chunk_start:chunk_end] = process_sequences(
            sequences, model, batch_size
        )
    
    smoother = ViterbiF0Smoother()
    smoothed_predictions = smoother.smooth(predictions, original_f0)
    
    return time_stamps, smoothed_predictions, original_f0

def extract_f0_for_prediction(
    audio_path: str,
    time_step: float = 0.01,
    pitch_floor: float = 75,
    pitch_ceiling: float = 600
) -> Tuple[np.ndarray, np.ndarray]:
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch(
        time_step=time_step,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling
    )
    pitch_values = pitch.selected_array['frequency']
    time_stamps = pitch.xs()
    pitch_values[pitch_values == 0] = 0
    return time_stamps, pitch_values

def create_sequences(
    original_f0: np.ndarray,
    chunk_start: int,
    chunk_length: int,
    sequence_length: int
) -> np.ndarray:
    sequences = np.zeros((chunk_length, sequence_length))
    for i in range(chunk_length):
        start_idx = max(0, chunk_start + i - sequence_length // 2)
        end_idx = start_idx + sequence_length
        if end_idx > len(original_f0):
            start_idx = len(original_f0) - sequence_length
            end_idx = len(original_f0)
        sequences[i] = original_f0[start_idx:end_idx]
    return sequences

def process_sequences(
    sequences: np.ndarray,
    model: F0PredictionModel,
    batch_size: int
) -> np.ndarray:
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(
                next(model.parameters()).device
            )
            batch_predictions = model(batch_tensor)
            predictions.extend(batch_predictions.cpu().numpy())
    
    return np.array(predictions).flatten()