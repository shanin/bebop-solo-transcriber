import numpy as np
import pandas as pd
import os
import argparse

# Calculate spectral flux from separated WAV audio
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import SpectrogramProcessor
from madmom.features.onsets import spectral_flux


def calculate_spectral_flux(audio_path, fps=100, frame_size=2048, hop_size=512):
    """
    Calculate spectral flux from a WAV audio file.
    
    Parameters:
    -----------
    audio_path : str
        Path to the WAV audio file
    fps : int, optional
        Frames per second for the analysis
    frame_size : int, optional
        Size of the analysis window in samples
    hop_size : int, optional
        Number of samples between consecutive frames
        
    Returns:
    --------
    flux : numpy.ndarray
        Spectral flux values
    """
    # Create processing chain
    sig = SignalProcessor(num_channels=1)
    frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size, fps=fps)
    stft = ShortTimeFourierTransformProcessor()
    spec = SpectrogramProcessor()
    
    # Process the audio
    signal = sig(audio_path)
    frames = frames(signal)
    stft = stft(frames)
    spectrogram = spec(stft)
    
    # Calculate spectral flux
    flux = spectral_flux(spectrogram)
    
    return flux

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_list", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    data_list = pd.read_csv(args.data_list)
    for _, row in data_list.iterrows():
        example_id = row['example_id']
        audio_path = row['clean_solo']
        flux = calculate_spectral_flux(audio_path)
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        np.save(os.path.join(args.output_folder, example_id + '.flux.npy'), flux)