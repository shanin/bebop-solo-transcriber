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
    def __init__(self, data_dir=None, source: str = 'original', split: str = 'all'):
        self.data_dir = data_dir
        self.source = source
        self.rhythm_tokens = RHYTHM_TOKENS
        self.all_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.pt')]
        self.split = split
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
        elif self.split == 'all':
            self.files = self.all_files
        else:
            assert False, f"Invalid split: {self.split}"

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(file_path)
        data = self.process_annotations(data)
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
        self.train_files = [f for f in self.all_files if f not in self.test_files and f not in self.val_files]


class FilosaxDataset(TrackDataset):
    def __init__(self, data_dir=None, source: str = 'original', split: str = 'all'):
        super().__init__(data_dir, source, split)
        self.instrument = 'tenor'

    def prepare_splits(self):
        self.test_files = [f'FS{i}_46.{self.source}.pt' for i in range(1, 6)] + \
                    [f'FS{i}_47.{self.source}.pt' for i in range(1, 6)] + \
                    [f'FS{i}_48.{self.source}.pt' for i in range(1, 6)]
        self.val_files = [f'FS{i}_45.{self.source}.pt' for i in range(1, 6)]
        self.train_files = [f for f in self.all_files if f not in self.test_files and f not in self.val_files and f.endswith(f'.{self.source}.pt')]

"""
class SegmentDataset(Dataset):
    def __init__(self, dataset, num_consecutive_bars: int, random_transposition: bool = False):
        self.dataset = dataset
        self.num_consecutive_bars = num_consecutive_bars
        self.random_transposition = random_transposition
"""
    


class DEPRECATED_ConsecutiveBarSampler(Sampler):
    def __init__(self, dataset, num_consecutive_bars: int, batch_size: int, inference: bool = False):
        self.dataset = dataset
        self.num_consecutive_bars = num_consecutive_bars
        self.batch_size = batch_size
        self.inference = inference
        
        if self.inference:
            self.starts = np.arange(len(self.dataset) - self.num_consecutive_bars + 1)[::self.num_consecutive_bars]
        else:
            self.starts = np.arange(len(self.dataset) - self.num_consecutive_bars + 1)
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle sequences
        if not self.inference:
            np.random.shuffle(self.starts)
        
        # Yield batches of sequences
        for i in range(0, len(self.starts), self.batch_size):
            batch_starts = self.starts[i:i + self.batch_size]
            # Flatten the batch sequences into a single list
            yield [idx for start in batch_starts for idx in range(start, start + self.num_consecutive_bars)]
    
    def __len__(self) -> int:
        return (len(self.starts) + self.batch_size - 1) // self.batch_size
    


def DEPRECATED_collate_fn(batch, num_consecutive_bars, random_transposition: bool = False, mode: str = 'tenor'):
    # Calculate actual batch size from the input
    actual_batch_size = len(batch) // num_consecutive_bars

    # Stack features
    features = torch.stack([item['features'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 48, 3)
    
    # Stack activations
    activations = torch.stack([item['activations'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 48, 384)
    
    # Additional NaN filtering at batch level
    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
    activations = torch.nan_to_num(activations, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Stack tokens
    tokens = torch.stack([item['tokens'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 48)
    mask = torch.stack([item['mask'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 48)
    inferred_time_feel = torch.stack([item['inferred_time_feel'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 4)
    source_time_feel = torch.stack([item['source_time_feel'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 1)
    
    # Convert rhythm tokens to tensors and stack
    rhythm_tokens = torch.stack([item['rhythm_tokens'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 4)
    
    if random_transposition:
        if mode == 'tenor':
            min_shift = -3
            max_shift = 9
        elif mode == 'alto':
            min_shift = -8
            max_shift = 4
        else:
            raise ValueError(f"Invalid mode: {mode}")
        for i in range(actual_batch_size):
            # Get all pitch tokens (0-127) for this batch item
            pitch_mask = (tokens[i] < 128)
            if pitch_mask.any():
                min_pitch = tokens[i][pitch_mask].min().item()
                max_pitch = tokens[i][pitch_mask].max().item()
                min_shift = max(min_shift, - min_pitch)
                max_shift = min(max_shift, 127 - max_pitch)
                shift = np.random.randint(min_shift, max_shift + 1)
                # Apply shift only to pitch tokens
                pitch_tokens = tokens[i][pitch_mask]
                tokens[i][pitch_mask] = pitch_tokens + shift
                activations[i] = torch.roll(activations[i], shifts=shift*3, dims=-1)

    return {
        'x': {
            'features': features,
            'activations': activations,
            'bar_num': torch.arange(num_consecutive_bars).unsqueeze(0).expand(actual_batch_size, -1)
        },
        'y': {
            'tokens': tokens,
            'rhythm_tokens': rhythm_tokens,
            'mask': mask,
            'inferred_time_feel': inferred_time_feel,
            'source_time_feel': source_time_feel
        }
    }

def DEPRECATED_dataloader_generator(dataset,  
                        num_consecutive_bars: int,
                        random_transposition: bool = False,
                        inference: bool = False,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        mode: str = 'tenor') -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader that samples consecutive bars.
    
    Args:
        dataset: The dataset to create a loader for
        num_consecutive_bars: Number of consecutive bars to sample
        batch_size: Number of sequences per batch
        shuffle: Whether to shuffle the sequences
    
    Returns:
        DataLoader that yields batches of consecutive bars
    """
    if not inference:
        sampler = ConsecutiveBarSampler(dataset, num_consecutive_bars=num_consecutive_bars, batch_size=batch_size, inference=inference)
        collate_local_fn = partial(
            collate_fn, 
            num_consecutive_bars=num_consecutive_bars,
            random_transposition=random_transposition,
            mode=mode,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_local_fn,
            pin_memory=True,             # speeds up host‑to‑device transfers
    persistent_workers=True,  
        )