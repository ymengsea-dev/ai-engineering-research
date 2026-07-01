## Raw Waveform-to-Spectrogram Engine

### The Goal

Your objective is to write an audio feature extraction script using PyTorch and its dedicated audio library, Torchaudio. You will generate a synthetic compound sound wave (simulating an audio clip containing both a low bass drone and a high-pitched whistle), analyze its signals, and convert it into a visual 2D Mel-Spectrogram tensor.

### The Blueprint & Instructions

Make sure you have torchaudio installed in your Python environment (pip install torchaudio).

- Synthesize an Audio Waveform: Combine a low-frequency sine wave (e.g., 400 Hz) and a high-frequency sine wave (e.g., 4000 Hz) into a single 1D tensor to simulate real complex noise.

- Apply Feature Extraction: Pass your synthetic raw waveform directly through torchaudio.transforms.MelSpectrogram.

- Verify Matrix Dimensions: Assert that your 1D wave has successfully transformed into a 2D acoustic map of shape (Frequency Bins, Time Steps).

### Test Harness & Expected Results

```
import torch
import torchaudio
import torchaudio.transforms as T

# --- DO NOT ALTER THIS DATA SETUP ---
torch.manual_seed(42)
sample_rate = 16000  # 16 kHz standard
duration_seconds = 2 # 2 seconds of audio

# Generate time steps
t = torch.linspace(0, duration_seconds, steps=sample_rate * duration_seconds)

# Create a complex sound: a low pitch (400Hz) combined with a high pitch (4000Hz)
wave_low = torch.sin(2 * torch.pi * 400 * t)
wave_high = torch.sin(2 * torch.pi * 4000 * t)
mock_waveform = (wave_low + wave_high).unsqueeze(0)  # Shape: (Channels=1, Samples=32000)
# ------------------------------------


# --- TODO: YOUR IMPLEMENTATION HERE ---
# Task 1: Initialize a MelSpectrogram transformation layer from torchaudio.
# Use these exact arguments: sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=64
mel_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=400,
    hop_length=160,
    n_mels=64
)

# Task 2: Pass 'mock_waveform' through your transform layer to get your features matrix
mel_spectrogram = mel_transform(mock_waveform)


# --- AUTOMATED PROFESSOR VERIFICATION SYSTEM ---
print("--- PHASE VI SPEECH & AUDIO VERIFICATION ---")
print(f"Raw Input Waveform Shape (Channels, Samples): {mock_waveform.shape}")
print(f"Processed Mel-Spectrogram Shape (Channels, Mels, Time): {mel_spectrogram.shape}")

# Extraction check
channels, n_mels, time_steps = mel_spectrogram.shape
assert n_mels == 64, f"Expected 64 mel frequency bins, got {n_mels}"
assert time_steps == 201, f"Expected 201 temporal step snapshots, got {time_steps}"

print("\nSuccess! Your digital signal analysis engine has successfully mapped audio tensors.")
```

### Expected Target Outputs to Match

- Raw Input Waveform Shape: torch.Size([1, 32000]) (1 mono channel, containing 32,000 numeric pressure points spanning 2 seconds).

- Processed Mel-Spectrogram Shape: torch.Size([1, 64, 201]) (1 mono channel, completely compressed into 64 distinct auditory frequency tracks across 201 continuous time snapshots).