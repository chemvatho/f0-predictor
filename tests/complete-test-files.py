# tests/test_model.py
import pytest
import torch
import numpy as np
from f0_predictor.model import F0PredictionModel
from f0_predictor.smoother import ViterbiF0Smoother

def test_model_initialization():
    model = F0PredictionModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        sequence_length=50
    )
    assert isinstance(model, F0PredictionModel)

def test_model_forward():
    model = F0PredictionModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        sequence_length=50
    )
    
    # Create dummy input
    batch_size = 32
    sequence_length = 50
    input_size = 1
    x = torch.randn(batch_size, sequence_length, input_size)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()

def test_viterbi_smoother():
    smoother = ViterbiF0Smoother(max_jump=35.0, voicing_threshold=0.5)
    
    # Create dummy predictions and original F0
    N = 100
    predictions = np.random.uniform(100, 300, N)
    original_f0 = np.random.uniform(100, 300, N)
    
    # Apply smoothing
    smoothed = smoother.smooth(predictions, original_f0)
    
    # Check output
    assert len(smoothed) == N
    assert not np.isnan(smoothed).any()
    
    # Check if values are within expected range
    assert np.all((smoothed == 0) | ((smoothed >= 50) & (smoothed <= 600)))

def test_model_input_validation():
    model = F0PredictionModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        sequence_length=50
    )
    
    # Test with wrong sequence length
    with pytest.raises(RuntimeError):
        wrong_sequence = torch.randn(32, 30, 1)  # Wrong sequence length (30 instead of 50)
        model(wrong_sequence)
    
    # Test with wrong input size
    with pytest.raises(RuntimeError):
        wrong_input = torch.randn(32, 50, 2)  # Wrong input size (2 instead of 1)
        model(wrong_input)

def test_smoother_transition_probability():
    smoother = ViterbiF0Smoother()
    
    # Test unvoiced to unvoiced transition
    prob_uu = smoother._calculate_transition_probability(0, 0)
    assert prob_uu == 0.5
    
    # Test voiced to unvoiced transition
    prob_vu = smoother._calculate_transition_probability(100, 0)
    assert prob_vu == 0.1
    
    # Test voiced to voiced transition
    prob_vv = smoother._calculate_transition_probability(100, 110)
    assert 0 < prob_vv <= 1

def test_smoother_emission_probability():
    smoother = ViterbiF0Smoother()
    
    # Test unvoiced observation
    prob_u = smoother._calculate_emission_probability(0, 0)
    assert prob_u == 0.1
    
    # Test voiced observation
    prob_v = smoother._calculate_emission_probability(100, 100)
    assert prob_v > 0
    
    # Test mismatch between prediction and observation
    prob_m = smoother._calculate_emission_probability(100, 200)
    assert prob_m < prob_v