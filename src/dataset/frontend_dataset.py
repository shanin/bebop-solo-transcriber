from torch.utils.data import Dataset
import torch

import os
import json
from tqdm import tqdm
from torch.utils.data import Sampler
from typing import Iterator, List
import numpy as np
from functools import partial

class FrontendTrackDataset(Dataset):
    def __init__(self, data_dir_x=None, data_dir_y=None, split: str = 'all', min_pitch_shift: int = 0, max_pitch_shift: int = 0):
        self.data_dir_x = data_dir_x
        self.data_dir_y = data_dir_y
        self.min_pitch_shift = min_pitch_shift
        self.max_pitch_shift = max_pitch_shift

        #self.all_x_files = [f for f in sorted(os.listdir(data_dir_x)) if f.endswith(f'.npy') and '.melspec' in f]
        self.all_y_files = [f for f in sorted(os.listdir(data_dir_y)) if f.endswith(f'.frame_labels.npy')]
        self.all_x_files = [f.replace('.frame_labels.npy', f'.melspec.{j}.npy') for f in self.all_y_files for j in range(self.min_pitch_shift, self.max_pitch_shift + 1)]
        self.all_pitch_shifts = [j for _ in self.all_y_files for j in range(self.min_pitch_shift, self.max_pitch_shift + 1)]

        self.split = split
        self.instrument = 'none'
        if self.split != 'all':
            self.prepare_splits()
        self.prepare_file_list()

    def prepare_file_list(self):
        if self.split == 'train':
            self.files_x = self.train_files_x
            self.files_y = self.train_files_y
            self.pitch_shifts = self.train_pitch_shifts
        elif self.split == 'val':
            self.files_x = self.val_files_x
            self.files_y = self.val_files_y
            self.pitch_shifts = self.val_pitch_shifts
        elif self.split == 'test':
            self.files_x = self.test_files_x
            self.files_y = self.test_files_y
            self.pitch_shifts = self.test_pitch_shifts
        elif self.split == 'all':
            self.files_x = self.all_x_files 
            self.files_y = self.all_y_files
            self.pitch_shifts = self.all_pitch_shifts
        else:
            assert False, f"Invalid split: {self.split}"

    def __len__(self):
        return len(self.files_x)

    def perform_pitch_shift(self, y, shift):
        y['onset'] = np.roll(y['onset'], shift)
        y['offset'] = np.roll(y['offset'], shift)
        y['frames'] = np.roll(y['frames'], shift)
        return y

    def __getitem__(self, idx):
        file_path_x = os.path.join(self.data_dir_x, self.files_x[idx])
        file_path_y = os.path.join(self.data_dir_y, self.files_y[idx])
        x = np.load(file_path_x, allow_pickle=True)
        y = np.load(file_path_y, allow_pickle=True).item()
        file_x = file_path_x.split('/')[-1].split('.')[0]
        file_y = file_path_y.split('/')[-1].split('.')[0]
        assert file_x == file_y, f"File names do not match: {file_x} and {file_y}"
        length = min(x.shape[0], y['onset'].shape[0], y['offset'].shape[0], y['frames'].shape[0])
        x = x[:length]
        y['onset'] = y['onset'][:length]
        y['offset'] = y['offset'][:length]
        y['frames'] = y['frames'][:length]

        if self.pitch_shifts[idx] != 0:
            y = self.perform_pitch_shift(y, self.pitch_shifts[idx])

        return {
            'mel_spec': x,
            'onset': y['onset'],
            'offset': y['offset'],
            'frames': y['frames'],
            'id': self.files_x[idx].split('.')[0]
        }

class FilosaxFrontendTrackDataset(FrontendTrackDataset):
    def prepare_splits(self):
        self.test_files_x = [f'FS{i}_46.melspec.0.npy' for i in range(1, 6)] + \
                    [f'FS{i}_47.melspec.0.npy' for i in range(1, 6)] + \
                    [f'FS{i}_48.melspec.0.npy' for i in range(1, 6)]
        self.val_files_x = [f'FS{i}_45.melspec.0.npy' for i in range(1, 6)]
        self.train_files_x = [f'FS{i}_{j:02d}.melspec.{k}.npy' for i in range(1, 6) for j in range(1, 46) for k in range(self.min_pitch_shift, self.max_pitch_shift + 1)]
        self.test_files_y = [f'FS{i}_46.frame_labels.npy' for i in range(1, 6)] + \
                    [f'FS{i}_47.frame_labels.npy' for i in range(1, 6)] + \
                    [f'FS{i}_48.frame_labels.npy' for i in range(1, 6)]
        self.val_files_y = [f'FS{i}_45.frame_labels.npy' for i in range(1, 6)]
        self.train_files_y = [f'FS{i}_{j:02d}.frame_labels.npy' for i in range(1, 6) for j in range(1, 46) for _ in range(self.min_pitch_shift, self.max_pitch_shift + 1)]
        self.train_pitch_shifts = [k for i in range(1, 6) for j in range(1, 46) for k in range(self.min_pitch_shift, self.max_pitch_shift + 1)]
        self.val_pitch_shifts = [0 for i in range(5)]
        self.test_pitch_shifts = [0 for i in range(15)]



class FrontendSegmentDataset(Dataset):
    def __init__(self, dataset, num_consecutive_frames: int, overlap_frames: int = 50, use_cache: bool = True):
        """
        Args:
            dataset: FrontendTrackDataset object
            num_consecutive_frames: int, length of each segment
            overlap_frames: int, overlap between consecutive segments (default: 50)
            use_cache: bool
        """
        self.dataset = dataset
        self.num_consecutive_frames = num_consecutive_frames
        self.overlap_frames = overlap_frames
        self.index = []
        self.cache = {}
        self.use_cache = use_cache
        for track_idx, track in enumerate(self.dataset):
            num_frames = track['mel_spec'].shape[0]
            
            # Calculate step size for overlapping segments
            step_size = self.num_consecutive_frames - self.overlap_frames
            
            # Start with padding: first segment starts at negative index
            start_offset = -(self.overlap_frames // 2)
            
            # Include all full segments with overlap, starting from the padded position
            for i in range(start_offset, num_frames - self.num_consecutive_frames + 1, step_size):
                self.index.append((track_idx, i))
            
            # Include tail segment if the last segment doesn't cover the end
            if len(self.index) == 0 or self.index[-1][1] + self.num_consecutive_frames < num_frames:
                # Add a final segment that ends at num_frames
                tail_start = max(start_offset, num_frames - self.num_consecutive_frames)
                # Only add if it's not the same as the last segment
                if len(self.index) == 0 or self.index[-1][1] != tail_start:
                    self.index.append((track_idx, tail_start))
            if self.use_cache:
                self.cache[track_idx] = track
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        track_idx, frame_idx = self.index[idx]
        if self.use_cache:
            if track_idx not in self.cache:
                self.cache[track_idx] = self.dataset[track_idx]
            track = self.cache[track_idx]
        else:
            track = self.dataset[track_idx]
        
        # Handle negative frame_idx (padding at the beginning)
        if frame_idx < 0:
            # We need to pad at the beginning
            pad_start = -frame_idx
            actual_start = 0
            actual_end = min(self.num_consecutive_frames - pad_start, track['mel_spec'].shape[0])
        else:
            pad_start = 0
            actual_start = frame_idx
            actual_end = frame_idx + self.num_consecutive_frames
        
        # Extract the actual data portion
        if actual_end > actual_start:
            mel_spec = track['mel_spec'][actual_start:actual_end]
            onset = track['onset'][actual_start:actual_end]
            offset = track['offset'][actual_start:actual_end]
            frames = track['frames'][actual_start:actual_end]
        else:
            # Edge case: no actual data to extract
            mel_spec = np.empty((0, track['mel_spec'].shape[1]), dtype=track['mel_spec'].dtype)
            onset = np.empty((0, track['onset'].shape[1]), dtype=track['onset'].dtype)
            offset = np.empty((0, track['offset'].shape[1]), dtype=track['offset'].dtype)
            frames = np.empty((0, track['frames'].shape[1]), dtype=track['frames'].dtype)
        
        # Add padding at the beginning if needed
        if pad_start > 0:
            # Pad mel_spec with -100 at the beginning
            mel_spec_pad_shape = list(mel_spec.shape)
            mel_spec_pad_shape[0] = pad_start
            mel_spec_pad = np.full(mel_spec_pad_shape, -100.0, dtype=mel_spec.dtype)
            mel_spec = np.concatenate([mel_spec_pad, mel_spec], axis=0)
            
            # Pad onset, offset, frames with 0 at the beginning
            onset_pad_shape = list(onset.shape)
            onset_pad_shape[0] = pad_start
            onset_pad = np.zeros(onset_pad_shape, dtype=onset.dtype)
            onset = np.concatenate([onset_pad, onset], axis=0)
            
            offset_pad_shape = list(offset.shape)
            offset_pad_shape[0] = pad_start
            offset_pad = np.zeros(offset_pad_shape, dtype=offset.dtype)
            offset = np.concatenate([offset_pad, offset], axis=0)
            
            frames_pad_shape = list(frames.shape)
            frames_pad_shape[0] = pad_start
            frames_pad = np.zeros(frames_pad_shape, dtype=frames.dtype)
            frames = np.concatenate([frames_pad, frames], axis=0)
        
        # Check if padding is needed at the end (for tail segments)
        actual_length = mel_spec.shape[0]
        if actual_length < self.num_consecutive_frames:
            pad_length = self.num_consecutive_frames - actual_length
            
            # Pad mel_spec with -100
            mel_spec_shape = list(mel_spec.shape)
            mel_spec_shape[0] = pad_length
            mel_spec_pad = np.full(mel_spec_shape, -100.0, dtype=mel_spec.dtype)
            mel_spec = np.concatenate([mel_spec, mel_spec_pad], axis=0)
            
            # Pad onset, offset, frames with 0
            onset_shape = list(onset.shape)
            onset_shape[0] = pad_length
            onset_pad = np.zeros(onset_shape, dtype=onset.dtype)
            onset = np.concatenate([onset, onset_pad], axis=0)
            
            offset_shape = list(offset.shape)
            offset_shape[0] = pad_length
            offset_pad = np.zeros(offset_shape, dtype=offset.dtype)
            offset = np.concatenate([offset, offset_pad], axis=0)
            
            frames_shape = list(frames.shape)
            frames_shape[0] = pad_length
            frames_pad = np.zeros(frames_shape, dtype=frames.dtype)
            frames = np.concatenate([frames, frames_pad], axis=0)
        
        segment = {
            'x': {
                'mel_spec': mel_spec,
            },
            'y': {
                'onset': onset,
                'offset': offset,
                'frames': frames,
            },
        }
        return segment



class FrontendInferenceSegmentDataset(Dataset):
    def __init__(self, dataset, num_consecutive_frames: int, overlap_frames: int = 50, use_cache: bool = True):
        """
        Args:
            dataset: FrontendTrackDataset object
            num_consecutive_frames: int, length of each segment
            overlap_frames: int, overlap between consecutive segments (default: 50)
            use_cache: bool
        """
        self.dataset = dataset
        self.num_consecutive_frames = num_consecutive_frames
        self.overlap_frames = overlap_frames
        self.index = []
        self.cache = {}
        self.use_cache = use_cache
        for track_idx, track in enumerate(self.dataset):
            num_frames = track['mel_spec'].shape[0]
            
            # Calculate step size for overlapping segments
            step_size = self.num_consecutive_frames - self.overlap_frames
            
            # Start with padding: first segment starts at negative index
            start_offset = -(self.overlap_frames // 2)
            
            # Include all full segments with overlap, starting from the padded position
            for i in range(start_offset, num_frames - self.num_consecutive_frames + 1, step_size):
                self.index.append((track_idx, i))
            
            # Include tail segment if the last segment doesn't cover the end
            if len(self.index) == 0 or self.index[-1][1] + self.num_consecutive_frames < num_frames:
                # Add a final segment that ends at num_frames
                tail_start = max(start_offset, num_frames - self.num_consecutive_frames)
                # Only add if it's not the same as the last segment
                if len(self.index) == 0 or self.index[-1][1] != tail_start:
                    self.index.append((track_idx, tail_start))
            if self.use_cache:
                self.cache[track_idx] = track
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        track_idx, frame_idx = self.index[idx]
        if self.use_cache:
            if track_idx not in self.cache:
                self.cache[track_idx] = self.dataset[track_idx]
            track = self.cache[track_idx]
        else:
            track = self.dataset[track_idx]
        
        # Handle negative frame_idx (padding at the beginning)
        if frame_idx < 0:
            # We need to pad at the beginning
            pad_start = -frame_idx
            actual_start = 0
            actual_end = min(self.num_consecutive_frames - pad_start, track['mel_spec'].shape[0])
        else:
            pad_start = 0
            actual_start = frame_idx
            actual_end = frame_idx + self.num_consecutive_frames
        
        # Extract the actual data portion
        if actual_end > actual_start:
            mel_spec = track['mel_spec'][actual_start:actual_end]
        else:
            # Edge case: no actual data to extract
            mel_spec = np.empty((0, track['mel_spec'].shape[1]), dtype=track['mel_spec'].dtype)
        
        # Add padding at the beginning if needed
        if pad_start > 0:
            # Pad mel_spec with -100 at the beginning
            mel_spec_pad_shape = list(mel_spec.shape)
            mel_spec_pad_shape[0] = pad_start
            mel_spec_pad = np.full(mel_spec_pad_shape, -100.0, dtype=mel_spec.dtype)
            mel_spec = np.concatenate([mel_spec_pad, mel_spec], axis=0)
            

        # Check if padding is needed at the end (for tail segments)
        actual_length = mel_spec.shape[0]
        if actual_length < self.num_consecutive_frames:
            pad_length = self.num_consecutive_frames - actual_length
            
            # Pad mel_spec with -100
            mel_spec_shape = list(mel_spec.shape)
            mel_spec_shape[0] = pad_length
            mel_spec_pad = np.full(mel_spec_shape, -100.0, dtype=mel_spec.dtype)
            mel_spec = np.concatenate([mel_spec, mel_spec_pad], axis=0)

        
        segment = {
            'x': {
                'mel_spec': mel_spec,
            },
        }
        return segment