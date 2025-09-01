import os
import numpy as np
import argparse
import json

from src.dataset.frontend_dataset import FrontendInferenceSegmentDataset
from src.model.crnn_frontend import MusicTranscriptionLightning


import torch
from torch.utils.data import DataLoader


from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import librosa
import torchaudio

import pyloudnorm as pyln


def loudnorm(x, sr, target_lufs=-23.0):
    peak_normalized_audio = pyln.normalize.peak(x, -1.0)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(peak_normalized_audio)
    loudness_normalized_audio = pyln.normalize.loudness(peak_normalized_audio, loudness, target_lufs)
    return loudness_normalized_audio


def extract_melspec(args, audio_path, fragment_start, fragment_end):
    print(f"Extracting melspec from {audio_path} from {fragment_start} to {fragment_end}")
    
    hop_length = args.sample_rate // args.frames_per_second
    f_max = args.f_max if args.f_max is not None else args.sample_rate // 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    x, sr = torchaudio.load(audio_path)
    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    if fragment_end is not None:
        x = x[:,fragment_start*sr : fragment_end*sr]
    else:
        x = x[:,fragment_start*sr:]

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

    del x, spectrogram, logmel
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return logmel_numpy


def frontend_inference(args, audio_path, frontend_checkpoint_path):
    print(f"Inferring onsets, offsets and frames from {audio_path}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = MusicTranscriptionLightning.load_from_checkpoint(frontend_checkpoint_path)
    model.eval()
    model.to(device)

    melspec = extract_melspec(args, audio_path, args.fragment_start, args.fragment_end)
    track = {'mel_spec': melspec}
    segments = FrontendInferenceSegmentDataset([track], num_consecutive_frames = args.frames, use_cache = False, overlap_frames = args.overlap)
    loader = DataLoader(segments, batch_size=args.batch_size, shuffle=False, num_workers=0)
    onsets = []
    offsets = []
    frames = []
    for batch in loader:
        output = model(batch['x']['mel_spec'].to(device))
        onsets.append(output['reg_onset_output'].to('cpu').detach()[:,int(args.overlap/2):-int(args.overlap/2),:])
        offsets.append(output['reg_offset_output'].to('cpu').detach()[:,int(args.overlap/2):-int(args.overlap/2),:])
        frames.append(output['frame_output'].to('cpu').detach()[:,int(args.overlap/2):-int(args.overlap/2),:])
    frames = np.concatenate(frames).reshape(-1, 88)[:track['mel_spec'].shape[0]]
    onsets = np.concatenate(onsets).reshape(-1, 88)[:track['mel_spec'].shape[0]]
    offsets = np.concatenate(offsets).reshape(-1, 88)[:track['mel_spec'].shape[0]]
    return frames, onsets, offsets


if __name__ == '__main__':
    print(f"Starting RhythmPerceiver inference")

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to the audio file (separated audio)')
    parser.add_argument('--fragment_start', type=int, default=0, help='Start time of the fragment (in seconds)')
    parser.add_argument('--fragment_end', type=int, default=None, help='End time of the fragment (in seconds)')
    parser.add_argument('--output_dir', type=str, required=True)

    # melspec parameters
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

    # frontend parameters
    parser.add_argument('--frontend_checkpoint', type=str, required=True)
    parser.add_argument("--frames", type=int, default=500, help="Number of frames")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap between frames")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    file_list = os.listdir(args.folder)
    for file in file_list:
        if file.endswith('.wav') and '(Woodwinds)' in file:
            audio_path = os.path.join(args.folder, file)
            frames, onsets, offsets = frontend_inference(args, audio_path, args.frontend_checkpoint)
            np.save(os.path.join(args.output_dir, f'{file.replace(".wav", ".frames.npy")}'), frames)
            np.save(os.path.join(args.output_dir, f'{file.replace(".wav", ".onsets.npy")}'), onsets)
            np.save(os.path.join(args.output_dir, f'{file.replace(".wav", ".offsets.npy")}'), offsets)
    