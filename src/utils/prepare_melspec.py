import torchaudio
import os
import torch 
import numpy as np
import pandas as pd
import argparse
import pyloudnorm as pyln
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import librosa

def loudnorm(x, sr, target_lufs=-23.0):
  peak_normalized_audio = pyln.normalize.peak(x, -1.0)
  meter = pyln.Meter(sr)
  loudness = meter.integrated_loudness(peak_normalized_audio)
  loudness_normalized_audio = pyln.normalize.loudness(peak_normalized_audio, loudness, target_lufs)
  return loudness_normalized_audio


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_list", type=str, required=True)
  parser.add_argument("--output_folder", type=str, required=True)
  parser.add_argument("--recompute", type=bool, default=False)
  parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate for mel spectrogram")
  parser.add_argument("--frames_per_second", type=int, default=100, help="Frames per second for hop size calculation")
  parser.add_argument("--window_size", type=int, default=2048, help="FFT window size")
  parser.add_argument("--n_mels", type=int, default=229, help="Number of mel bands")
  parser.add_argument("--f_min", type=float, default=30.0, help="Minimum frequency")
  parser.add_argument("--f_max", type=float, default=None, help="Maximum frequency (defaults to sample_rate // 2)")
  parser.add_argument("--window", type=str, default='hann', help="Window function")
  parser.add_argument("--center", type=bool, default=True, help="Center the frames")
  parser.add_argument("--pad_mode", type=str, default='reflect', help="Padding mode")
  parser.add_argument("--ref", type=float, default=1.0, help="Reference value for dB conversion")
  parser.add_argument("--amin", type=float, default=1e-10, help="Minimum value for log")
  parser.add_argument("--top_db", type=float, default=None, help="Top dB for amplitude to dB conversion")
  parser.add_argument("--pitch_shift_min", type=float, default=0.0, help="Minimum pitch shift in semitones")
  parser.add_argument("--pitch_shift_max", type=float, default=0.0, help="Maximum pitch shift in semitones")
  args = parser.parse_args()

  os.makedirs(args.output_folder, exist_ok=True)

  # Calculate hop length based on frames per second
  hop_length = args.sample_rate // args.frames_per_second
  
  # Set f_max to Nyquist frequency if not specified
  f_max = args.f_max if args.f_max is not None else args.sample_rate // 2
  
  data_list = pd.read_csv(args.data_list)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Preparing mel spectrogram computation on device: ', device)
  print(f'Parameters: sample_rate={args.sample_rate}, window_size={args.window_size}, hop_length={hop_length}')
  print(f'Mel bins={args.n_mels}, fmin={args.f_min}, fmax={f_max}')
  
  # Initialize torchlibrosa extractors
  spectrogram_extractor = Spectrogram(
    n_fft=args.window_size, 
    hop_length=hop_length, 
    win_length=args.window_size, 
    window=args.window, 
    center=args.center, 
    pad_mode=args.pad_mode, 
    freeze_parameters=True
  ).to(device)
  
  logmel_extractor = LogmelFilterBank(
    sr=args.sample_rate, 
    n_fft=args.window_size, 
    n_mels=args.n_mels, 
    fmin=args.f_min, 
    fmax=f_max, 
    ref=args.ref, 
    amin=args.amin, 
    top_db=args.top_db, 
    freeze_parameters=True
  ).to(device)

  
  with torch.no_grad():
    for _, row in data_list.iterrows():
      for pitch_shift in range(args.pitch_shift_min, args.pitch_shift_max + 1):
        idx = row['example_id']
        output_file = os.path.join(args.output_folder, idx + f".melspec.{pitch_shift}.npy")
        
        if os.path.exists(output_file):
          if args.recompute:
            os.remove(output_file)
          else:
            print(f"Skipping {idx} - output already exists")
            continue

        print(f"Processing {idx} with pitch shift {pitch_shift}...")
        
        # Load audio
        x, sr = torchaudio.load(row['clean_solo'])
        
        # Convert to mono by averaging channels
        if x.shape[0] > 1:
          x = x.mean(dim=0, keepdim=True)
        
        # Apply pitch shift
        if pitch_shift != 0:
          x = librosa.effects.pitch_shift(x.squeeze().numpy(), sr, pitch_shift, bins_per_octave=12)
          x = torch.from_numpy(x).unsqueeze(0).to(device)

        # Apply loudnorm
        x_numpy = x.squeeze().numpy()
        x_normalized = loudnorm(x_numpy, sr)
        x = torch.from_numpy(x_normalized).unsqueeze(0).to(device)
        
        # Resample to target sample rate if needed
        if sr != args.sample_rate:
          resampler = torchaudio.transforms.Resample(sr, args.sample_rate).to(device)
          x = resampler(x)
        
        # Compute spectrogram using torchlibrosa
        spectrogram = spectrogram_extractor(x)
        
        # Convert to log mel spectrogram
        logmel = logmel_extractor(spectrogram)
        
        # Save mel spectrogram
        logmel_numpy = logmel.squeeze().cpu().numpy()
        np.save(output_file, logmel_numpy)
        
        print(f"Saved mel spectrogram for {idx} with shape: {logmel_numpy.shape}")

        # Clean up
        del x, spectrogram, logmel
        if torch.cuda.is_available():
          torch.cuda.empty_cache()