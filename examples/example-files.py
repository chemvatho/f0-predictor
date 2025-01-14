# examples/basic_usage.py
from f0_predictor import predict_f0_for_audio, plot_f0_comparison
from f0_predictor.model import F0PredictionModel

def main():
    # Initialize model
    model = F0PredictionModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        sequence_length=50
    )
    
    # Predict F0
    audio_path = "path/to/your/audio.wav"
    time_stamps, predictions, original_f0 = predict_f0_for_audio(
        audio_path=audio_path,
        model=model
    )
    
    # Plot results
    plot_f0_comparison(
        time_stamps=time_stamps,
        original_f0=original_f0,
        predicted_f0=predictions,
        save_path="f0_prediction.png"
    )

if __name__ == "__main__":
    main()