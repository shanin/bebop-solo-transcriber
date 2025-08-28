
from src.dataset.frontend_dataset import FrontendTrackDataset, FrontendSegmentDataset
from src.model.crnn_frontend import MusicTranscriptionLightning
from src.dataset.backend_dataset import BackendSegmentInferenceDataset, backend_inference_segment_collate_fn
from src.model.rhythm_perceiver import RhythmPerceiverLightningModule
import sys
import numpy as np
from src.dataset.frontend_dataset import FilosaxFrontendTrackDataset, FrontendSegmentDataset, FrontendTrackDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
import argparse
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import librosa
from madmom.features import (DBNDownBeatTrackingProcessor, RNNDownBeatProcessor) 
import os
import pyloudnorm as pyln
import torchaudio

model_name = 'best-model-epoch=12-val_loss=0.01.ckpt'
checkpoint_path = 'wandb_logs/bebop-solo-transcriber/819zvtw6/checkpoints/'

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

    x, sr = torchaudio.load(audio_path, frame_offset=fragment_start * args.sample_rate, num_frames=fragment_end * args.sample_rate)
    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)

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
    segments = FrontendSegmentDataset([track], num_consecutive_frames = args.frames, use_cache = False, overlap_frames = args.overlap)
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

def beat_tracking_inference(args):
    print(f"Inferring beats from {args.audio_background} from {args.fragment_start} to {args.fragment_end}")
    processor = DBNDownBeatTrackingProcessor(
        [4], 
        fps=args.fps, 
        min_bpm=int(args.bpm-15), 
        max_bpm=args.max_bpm, 
        transition_lambda=args.transition_lambda
    )
    in_processor = RNNDownBeatProcessor()

    y, sr = librosa.load(str(args.audio_background), sr=44100)
    y = y[int(args.fragment_start * sr):int(args.fragment_end * sr)]  # Remove first 5 seconds
    activations = in_processor(y)
    beats = processor(activations)

    return beats

def generate_posenc(features, beats):
    print(f"Generating position encoding from {features.shape[0]} frames and {beats.shape[0]} beats")
    # get bar_number
    bar_num = []
    beat_num = []
    beat_time = []
    current_bar = -1
    for elem in beats:
        if elem[1] == 1:
            current_bar += 1
        bar_num.append(current_bar)
        beat_num.append(elem[1] - 1)
        beat_time.append(elem[0])
    
    bar_index = [-1] * len(features)
    beat_index = [-1] * len(features)
    frame_phase = [-1] * len(features)
    frame_centers = np.arange(len(features)) / 100 + 0.005
    fourier = np.zeros((len(features), 12))

    for i in range(len(bar_num) - 1):
        mask = (frame_centers >= beat_time[i]) & (frame_centers < beat_time[i+1])
        for idx in np.where(mask)[0]:
            bar_index[idx] = bar_num[i]
            beat_index[idx] = beat_num[i]
            delta = (beat_time[i+1] - beat_time[i])
            phase = (frame_centers[idx] - beat_time[i]) / delta
            frame_phase[idx] = phase
            for j in range(6):
                fourier[idx, 2*j] = np.sin(2 * np.pi * phase * (j + 1))
                fourier[idx, 2*j+1] = np.cos(2 * np.pi * phase * (j + 1))
                
    bar_index = np.array(bar_index).reshape(-1, 1)
    beat_index = np.array(beat_index).reshape(-1, 1)
    frame_phase = np.array(frame_phase).reshape(-1, 1)
    posenc = np.concatenate([bar_index, beat_index, frame_phase, fourier], axis=-1)
    return posenc

def backend_inference(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frames, onsets, offsets = frontend_inference(args, args.audio_sax, args.frontend_checkpoint)
    beats = beat_tracking_inference(args)
    posenc = generate_posenc(frames, beats)
    track = {'onsets': onsets, 'offsets': offsets, 'frames': frames, 'posenc': posenc}
    segments = BackendSegmentInferenceDataset(track, num_consecutive_bars = args.num_consecutive_bars, use_cache = False)
    loader = DataLoader(segments, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=backend_inference_segment_collate_fn)
    print(f"Loading model from {args.backend_checkpoint}")
    model = RhythmPerceiverLightningModule.load_from_checkpoint(args.backend_checkpoint)
    model.eval()
    model.to(device)
    print(f"Running RhythmPerceiver")
    predictions = []
    for batch in loader:
        output = model(batch)
        current_prediction = model._generate_structured_predictions(output[0], output[1])
        predictions.append(current_prediction)
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_sax', type=str, required=True, help='Path to the audio file (separated audio)')
    parser.add_argument('--audio_background', type=str, required=True, help='Path to the audio file (background audio)')
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
    parser.add_argument('--frontend_checkpoint', type=str, required=True, default=checkpoint_path + model_name)
    parser.add_argument("--frames", type=int, default=500, help="Number of frames")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap between frames")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    # beat tracking parameters
    parser.add_argument('--bpm', type=int, default=150, help='Tempo of the audio')
    parser.add_argument('--max_bpm', type=int, default=350, help='Maximum tempo of the audio')
    parser.add_argument('--transition_lambda', type=int, default=1000, help='Transition lambda for the beat tracking')
    parser.add_argument('--fps', type=int, default=100, help='Frames per second for the beat tracking')

    # backend parameters
    parser.add_argument('--backend_checkpoint', type=str, required=True)
    parser.add_argument('--num_consecutive_bars', type=int, default=8, help='Number of consecutive bars')

    args = parser.parse_args()
    prediction = backend_inference(args)
    np.save(args.output_dir + '/prediction.npy', prediction)
    print(f"Saved prediction to {args.output_dir}/prediction.npy")