from torch.utils.data import Dataset
import torch
from src.tokenizer.rhythm_tokens import RHYTHM_TOKENS
import os
import json
from tqdm import tqdm
from torch.utils.data import Sampler
from typing import Iterator, List
import numpy as np
from functools import partial

class TrackDataset(Dataset):
    def __init__(self, data_dir=None, source: str = 'original', split: str = 'all', hard_transpose: int = 0):
        self.data_dir = data_dir
        self.source = source
        self.hard_transpose = hard_transpose
        self.rhythm_tokens = RHYTHM_TOKENS
        self.all_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.pt')]
        self.split = split
        self.instrument = 'none'
        if self.split != 'all':
            self.prepare_splits()
        self.prepare_file_list()

    def process_annotations(self, data):
        data['scalar_features'] = torch.nan_to_num(data['scalar_features'], nan=0.0, posinf=1.0, neginf=0.0)
        data['activations'] = torch.nan_to_num(data['activations'], nan=0.0, posinf=1.0, neginf=0.0)
        return data

    def prepare_file_list(self):
        if self.split == 'train':
            self.files = self.train_files
        elif self.split == 'val':
            self.files = self.val_files
        elif self.split == 'test':
            self.files = self.test_files
        elif self.split == 'xr_test':
            self.files = self.xr_test_files
        elif self.split == 'all':
            self.files = self.all_files
        else:
            assert False, f"Invalid split: {self.split}"

    def __len__(self):
        return len(self.files)

    def perform_hard_transposition(self, data):
        if self.hard_transpose != 0:
            pitch_mask = (data['tokens'] < 128)
            data['tokens'][pitch_mask] += self.hard_transpose
        return data

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path)
        data = self.process_annotations(data)
        data = self.perform_hard_transposition(data)
        return data
    

class OmnibookDataset(TrackDataset):
    def __init__(self, data_dir=None, source: str = 'original', split: str = 'all'):
        super().__init__(data_dir, source, split)
        self.instrument = 'alto'

    def prepare_splits(self):
        self.test_files = [
            'OB_1p64c.original.pt', 'OB_S5VYc.original.pt', 'OB_wkTyc.original.pt',
        ]
        self.val_files = [
            'OB_Nqn4c.original.pt', 'OB_6Cbwc.original.pt'
        ]
        self.xr_test_files = [f'OB_{f}.original.pt' for f in [
            'wv3wc', '3zn4c', 'D3fYc', 'nvJyc', 'myn4c', 'gRfYc',
            '1p64c', '7XTyc', 'N3fYc', '3RfYc', 'vRfYc', '2RfYc', 'mTHyc',
            'WRfYc', '9THwc', 'S1swc', 'KRfYc', 'Pq3yc', 'nRfYc', 'rRfYc',
            '7RfYc', 'S5VYc', '73bwc', 'N8swc', '6Cbwc', 'QRfYc', 'yp3wc',
            'lTXyc', 'cXbwc', 'LRfYc'
        ]]
        self.train_files = [f for f in self.all_files if f not in self.test_files and f not in self.val_files]

class FilosaxDataset(TrackDataset):
    def __init__(self, data_dir=None, source: str = 'original', split: str = 'all'):
        super().__init__(data_dir, source, split)
        self.instrument = 'tenor'
        self.hard_transpose = -14

    def prepare_splits(self):
        self.test_files = [f'FS{i}_46.{self.source}.pt' for i in range(1, 6)] + \
                    [f'FS{i}_47.{self.source}.pt' for i in range(1, 6)] + \
                    [f'FS{i}_48.{self.source}.pt' for i in range(1, 6)]
        self.val_files = [f'FS{i}_45.{self.source}.pt' for i in range(1, 6)]
        self.train_files = [f for f in self.all_files if f not in self.test_files and f not in self.val_files and f.endswith(f'.{self.source}.pt')]

class SegmentDataset(Dataset):
    def __init__(self, dataset, num_consecutive_bars: int, random_transposition: bool = False, use_cache: bool = True, pitch_shift: bool = False, disable_rhythm_classifier = False):
        """
        Args:
            dataset: TrackDataset object
            num_consecutive_bars: int
            random_transposition: bool
            use_cache: bool
            pitch_shift: bool - if True, the pitch of the segment is shifted by -1, 0 or 1 bin (1 semitone = 3 bins)
        """
        self.dataset = dataset
        self.mode = dataset.instrument
        self.num_consecutive_bars = num_consecutive_bars
        self.random_transposition = random_transposition
        self.pitch_shift = pitch_shift
        self.index = []
        self.cache = {}
        self.use_cache = use_cache
        self.disable_rhythm_classifier = disable_rhythm_classifier
        for track_idx, track in enumerate(self.dataset):
            num_bars = track['tokens'].shape[0]
            for i in range(0, num_bars - self.num_consecutive_bars + 1):
                self.index.append((track_idx, i))
            if self.use_cache:
                self.cache[track_idx] = track
    
    def __len__(self):
        return len(self.index)
    
    def transposition(self, segment):
        if self.mode == 'tenor':
            min_shift = -3
            max_shift = 9
        elif self.mode == 'alto':
            min_shift = -8
            max_shift = 4
        elif self.mode == 'none':
            min_shift = -3
            max_shift = 3
        pitch_mask = (segment['y']['tokens'] < 128)
        if pitch_mask.any():
            min_pitch = segment['y']['tokens'][pitch_mask].min().item()
            max_pitch = segment['y']['tokens'][pitch_mask].max().item()
            min_shift = max(min_shift, - min_pitch)
            max_shift = min(max_shift, 127 - max_pitch)
            shift = np.random.randint(min_shift, max_shift + 1)
            # Apply shift only to pitch tokens
            pitch_tokens = segment['y']['tokens'][pitch_mask]
            segment['y']['tokens'][pitch_mask] = pitch_tokens + shift
            segment['x']['activations'] = torch.roll(segment['x']['activations'], shifts=shift*3, dims=-1)
        return segment
    
    def apply_pitch_shift(self, segment):
        shift = np.random.randint(-1, 2)
        segment['x']['activations'] = torch.roll(segment['x']['activations'], shifts=shift, dims=-1)
        return segment

    def __getitem__(self, idx):
        track_idx, bar_idx = self.index[idx]
        if self.use_cache:
            if track_idx not in self.cache:
                self.cache[track_idx] = self.dataset[track_idx]
            track = self.cache[track_idx]
        else:
            track = self.dataset[track_idx]
        segment = {
            'x': {
                'activations': track['activations'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
                'features': track['scalar_features'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
            },
            'y': {
                'tokens': track['tokens'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
                'rhythm_tokens': track['rhythm_tokens'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
                'mask': track['mask'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
                'inferred_time_feel': track['inferred_time_feel'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
                'source_time_feel': track['source_time_feel'].clone(),
            },
            'meta': {
                'disable_rhythm_classifier': self.disable_rhythm_classifier,
            }
        }
        if self.random_transposition:
            segment = self.transposition(segment)
        if self.pitch_shift:
            segment = self.apply_pitch_shift(segment)
        return segment
    
class InferenceDataset(Dataset):
    def __init__(self, track, num_consecutive_bars: int):
        self.track = track
        self.num_consecutive_bars = num_consecutive_bars
        self.num_bars = self.track['scalar_features'].shape[0]
        self.full_segments = self.num_bars // self.num_consecutive_bars
        self.last_segment = self.num_bars % self.num_consecutive_bars
        self.num_segments = self.full_segments + (1 if self.last_segment > 0 else 0) # 1 if there is a last segment
    
    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        if idx >= self.num_segments:
            raise IndexError(f"Index {idx} is out of range for dataset with {self.num_segments} segments")
        if idx < self.full_segments:
            bar_idx = idx * self.num_consecutive_bars
        else:
            bar_idx = self.full_segments * self.num_consecutive_bars
            bar_idx -= (self.num_consecutive_bars - self.last_segment)

        segment = {
            'x': {
                'activations': self.track['activations'][bar_idx:bar_idx + self.num_consecutive_bars],
                'features': self.track['scalar_features'][bar_idx:bar_idx + self.num_consecutive_bars],
            },
        }
        return segment