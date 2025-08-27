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

class BackendTrackDataset(Dataset):
    def __init__(self, bars_dir=None, crnn_dir=None, posenc_dir=None, source: str = 'original', split: str = 'all', hard_transpose: int = 0):
        self.bars_dir = bars_dir
        self.crnn_dir = crnn_dir
        self.posenc_dir = posenc_dir
        self.source = source
        self.hard_transpose = hard_transpose
        self.rhythm_tokens = RHYTHM_TOKENS

        self.all_files = [f for f in sorted(os.listdir(bars_dir)) if f.endswith(f'.{source}.pt')]

        self.split = split
        self.instrument = 'none'
        if self.split != 'all':
            self.prepare_splits()
        self.prepare_file_list()


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
        file_path = os.path.join(self.bars_dir, self.files[idx])
        data = torch.load(file_path)
        data = self.perform_hard_transposition(data)

        onsets_file_path = os.path.join(self.crnn_dir, self.files[idx].replace(f'.{self.source}.pt', '.onsets.npy'))
        onsets_data = torch.from_numpy(np.load(onsets_file_path)).float()
        data['onsets'] = onsets_data

        offsets_file_path = os.path.join(self.crnn_dir, self.files[idx].replace(f'.{self.source}.pt', '.offsets.npy'))
        offsets_data = torch.from_numpy(np.load(offsets_file_path)).float()
        data['offsets'] = offsets_data

        frames_file_path = os.path.join(self.crnn_dir, self.files[idx].replace(f'.{self.source}.pt', '.frames.npy'))
        frames_data = torch.from_numpy(np.load(frames_file_path)).float()
        data['frames'] = frames_data

        posenc_file_path = os.path.join(self.posenc_dir, self.files[idx].replace(f'.{self.source}.pt', '.posenc.npy'))
        posenc_data = torch.from_numpy(np.load(posenc_file_path)).float()
        data['posenc'] = posenc_data

        return data

class FilosaxBackendDataset(BackendTrackDataset):
    def __init__(self, bars_dir=None, crnn_dir=None, posenc_dir=None, source: str = 'original', split: str = 'all'):
        super().__init__(bars_dir, crnn_dir, posenc_dir, source, split)
        self.instrument = 'tenor'
        self.hard_transpose = -14

    def prepare_splits(self):
        self.test_files = [f'FS{i}_46.{self.source}.pt' for i in range(1, 6)] + \
                    [f'FS{i}_47.{self.source}.pt' for i in range(1, 6)] + \
                    [f'FS{i}_48.{self.source}.pt' for i in range(1, 6)]
        self.val_files = [f'FS{i}_45.{self.source}.pt' for i in range(1, 6)]
        self.train_files = [f for f in self.all_files if f not in self.test_files and f not in self.val_files and f.endswith(f'.{self.source}.pt')]

class BackendSegmentDataset(Dataset):
    def __init__(self, dataset, num_consecutive_bars: int = 8, random_transposition: bool = False, use_cache: bool = True, pitch_shift: bool = False, disable_rhythm_classifier = False):
        """
        Args:
            dataset: BackendTrackDataset object
            num_consecutive_bars: int (default 8)
            random_transposition: bool
            use_cache: bool
            pitch_shift: bool - if True, the pitch of the segment is shifted by -1, 0 or 1 bin (1 semitone = 3 bins)
        """
        self.dataset = dataset
        self.num_consecutive_bars = num_consecutive_bars
        self.random_transposition = random_transposition
        self.pitch_shift = pitch_shift
        self.index = []
        self.cache = {}
        self.use_cache = use_cache
        self.disable_rhythm_classifier = disable_rhythm_classifier
        
        # Build index of all possible segments
        for track_idx, track in enumerate(self.dataset):
            num_bars = track['tokens'].shape[0]
            for i in range(0, num_bars - self.num_consecutive_bars + 1):
                self.index.append((track_idx, i))
            if self.use_cache:
                self.cache[track_idx] = track
    
    def __len__(self):
        return len(self.index)
    
    def create_bin_level_position_encoding(self, num_bars):
        """
        Create position encoding for bin-level and beat-level data.
        Bin latents: 12 per beat x 4 beats = 48 total
        Beat/Rhythm latents: 1 per beat x 4 beats = 4 total (separate from bins)
        
        Returns:
            bin_encoding: tensor of shape [num_bars, 48, 3] where last dim is [bar_idx, beat_idx, bin_idx]
            beat_encoding: tensor of shape [num_bars, 4, 3] where last dim is [bar_idx, beat_idx, 0]
        """
        bin_encoding = []
        beat_encoding = []
        
        for bar_idx in range(num_bars):
            bar_bin_encoding = []
            bar_beat_encoding = []
            
            for beat_idx in range(4):
                # 12 bins per beat (bin_idx 0-11)
                for bin_idx in range(12):
                    bar_bin_encoding.append([bar_idx, beat_idx, bin_idx])
                
                # Beat-level encoding (one per beat, separate from bins)
                bar_beat_encoding.append([bar_idx, beat_idx, 12])
            
            bin_encoding.append(bar_bin_encoding)
            beat_encoding.append(bar_beat_encoding)
        
        return torch.tensor(bin_encoding, dtype=torch.long), torch.tensor(beat_encoding, dtype=torch.long)
    
    def adjust_frame_level_posenc(self, posenc_data, segment_start_bar):
        """
        Adjust position encoding to start from 0 for this segment.
        
        Args:
            posenc_data: tensor of shape [total_frames, posenc_features] 
            segment_start_bar: int, the starting bar index for this segment
            
        Returns:
            Tuple of (concatenated_frames, adjusted_posenc)
        """
        # Adjust the absolute bar numbers in posenc to start from 0 for this segment
        adjusted_posenc = posenc_data.clone()
        if len(adjusted_posenc.shape) > 1 and adjusted_posenc.shape[1] > 0:
            adjusted_posenc[:, 0] = adjusted_posenc[:, 0] - segment_start_bar
        
        return adjusted_posenc
    
    def __getitem__(self, idx):
        track_idx, bar_idx = self.index[idx]
        if self.use_cache:
            if track_idx not in self.cache:
                self.cache[track_idx] = self.dataset[track_idx]
            track = self.cache[track_idx]
        else:
            track = self.dataset[track_idx]
        
        # Extract bin-level data (easy concatenation)
        bin_level_data = {
            'tokens': track['tokens'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
            'rhythm_tokens': track['rhythm_tokens'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
            'mask': track['mask'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
            'inferred_time_feel': track['inferred_time_feel'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
            'source_time_feel': track['source_time_feel'].clone(),
            'rhythm_signatures': track['rhythm_signatures'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
            'flags': track['flags'][bar_idx:bar_idx + self.num_consecutive_bars].clone(),
        }
        
        # Create bin-level position encoding
        bin_position_encoding, beat_position_encoding = self.create_bin_level_position_encoding(self.num_consecutive_bars)
        
        # Extract and process frame-level data
        # Find the frame indices corresponding to the selected bars
        # posenc contains [bar_idx, beat_idx, ...] in first two columns
        posenc_full = track['posenc']
        assert bar_idx >= 0, f"bar_idx is negative: {bar_idx}"
        # Assert that negative beat indices correspond to negative bar indices and vice versa
        beat_indices = posenc_full[:, 1]
        bar_indices = posenc_full[:, 0]
        assert torch.all((beat_indices < 0) == (bar_indices < 0)), f"Negative beat indices must correspond to negative bar indices: {beat_indices} {bar_indices}"
        bar_mask = (posenc_full[:, 0] >= bar_idx) & (posenc_full[:, 0] < bar_idx + self.num_consecutive_bars)
        
        frame_indices = torch.where(bar_mask)[0]
        if len(frame_indices) > 0:
            start_frame = frame_indices[0].item()
            end_frame = frame_indices[-1].item() + 1
            
            segment_onsets = track['onsets'][start_frame:end_frame]
            segment_offsets = track['offsets'][start_frame:end_frame]
            segment_frames = track['frames'][start_frame:end_frame]
            segment_posenc = track['posenc'][start_frame:end_frame]
            
            # Adjust position encoding for this segment
            adjusted_posenc = self.adjust_frame_level_posenc(segment_posenc, bar_idx)

        else:
            # Handle empty case
            segment_onsets = torch.empty(0, track['onsets'].shape[1] if len(track['onsets'].shape) > 1 else 0)
            segment_offsets = torch.empty(0, track['offsets'].shape[1] if len(track['offsets'].shape) > 1 else 0) 
            segment_frames = torch.empty(0, track['frames'].shape[1] if len(track['frames'].shape) > 1 else 0)
            adjusted_posenc = torch.empty(0, track['posenc'].shape[1] if len(track['posenc'].shape) > 1 else 0)
        
        frame_level_data = {
            'onsets': segment_onsets,
            'offsets': segment_offsets,
            'frames': segment_frames,
            'posenc': adjusted_posenc,
        }
        
        segment = {
            'bin_level': bin_level_data,
            'frame_level': frame_level_data,
            'bin_position_encoding': bin_position_encoding,
            'beat_position_encoding': beat_position_encoding,
            'meta': {
                'disable_rhythm_classifier': self.disable_rhythm_classifier,
                'track_idx': track_idx,
                'bar_idx': bar_idx,
            }
        }
        
        # Apply transformations if needed - not implemented yet
        #if self.random_transposition:
        #    segment = self.transposition(segment)
        #if self.pitch_shift:
        #    segment = self.apply_pitch_shift(segment)
            
        return segment
    
    def transposition(self, segment):
        # Apply transposition to bin-level data
        if hasattr(self.dataset, 'instrument') and self.dataset.instrument == 'tenor':
            # Apply random transposition logic here if needed
            pass
        return segment
    
    def apply_pitch_shift(self, segment):
        # Apply pitch shift to frame-level data if needed
        if self.pitch_shift:
            # Apply pitch shifting logic here if needed
            pass
        return segment


def backend_segment_collate_fn(batch):
    """
    Collate function for BackendSegmentDataset.
    
    Handles variable-length frame-level data by padding and creating masks.
    Bin-level data has consistent dimensions so can be stacked directly.
    
    Args:
        batch: List of segment dictionaries from BackendSegmentDataset
        
    Returns:
        Dictionary with batched data and masks for frame-level sequences
    """
    batch_size = len(batch)
    
    # Handle bin-level data (consistent dimensions) - just stack
    bin_level_keys = ['tokens', 'rhythm_tokens', 'mask', 'inferred_time_feel', 
                      'rhythm_signatures', 'flags']
    
    batched_bin_level = {}
    for key in bin_level_keys:
        if key in batch[0]['bin_level']:
            batched_bin_level[key] = torch.stack([item['bin_level'][key] for item in batch])
    
    # Handle source_time_feel separately (single value per track, not per segment)
    if 'source_time_feel' in batch[0]['bin_level']:
        batched_bin_level['source_time_feel'] = torch.stack([item['bin_level']['source_time_feel'] for item in batch])
    
    # Handle bin position encoding (consistent dimensions)
    bin_position_encoding = torch.stack([item['bin_position_encoding'] for item in batch])
    beat_position_encoding = torch.stack([item['beat_position_encoding'] for item in batch])
    
    # Handle frame-level data (variable dimensions) - need padding and masking
    frame_level_keys = ['onsets', 'offsets', 'frames', 'posenc']
    
    # Find maximum sequence length across the batch (should be same for all frame-level keys)
    max_frame_len = 0
    for item in batch:
        if frame_level_keys[0] in item['frame_level']:
            max_frame_len = max(max_frame_len, item['frame_level'][frame_level_keys[0]].shape[0])
    
    batched_frame_level = {}
    
    # Create single mask for all frame-level data (they share the same temporal dimension)
    frame_mask = torch.zeros(batch_size, max_frame_len, dtype=torch.bool)
    
    for key in frame_level_keys:
        if key in batch[0]['frame_level']:
            if max_frame_len == 0:
                # Handle empty case
                sample_shape = batch[0]['frame_level'][key].shape
                batched_frame_level[key] = torch.zeros(batch_size, 0, *sample_shape[1:])
                continue
                
            # Get feature dimensions from first non-empty sample
            feature_dims = None
            for item in batch:
                if item['frame_level'][key].shape[0] > 0:
                    feature_dims = item['frame_level'][key].shape[1:]
                    break
            
            if feature_dims is None:
                # All samples are empty
                batched_frame_level[key] = torch.zeros(batch_size, 0)
                continue
            
            # Create padded tensor
            padded_shape = (batch_size, max_frame_len, *feature_dims)
            padded_tensor = torch.zeros(padded_shape, dtype=batch[0]['frame_level'][key].dtype)
            
            # Fill in the data and create mask (only once, for the first key)
            for i, item in enumerate(batch):
                seq_len = item['frame_level'][key].shape[0]
                if seq_len > 0:
                    padded_tensor[i, :seq_len] = item['frame_level'][key]
                    # Only set mask once (on first key iteration)
                    if key == frame_level_keys[0]:
                        frame_mask[i, :seq_len] = True
            
            batched_frame_level[key] = padded_tensor
    
    # Handle metadata
    meta_data = {
        'disable_rhythm_classifier': [item['meta']['disable_rhythm_classifier'] for item in batch],
        'track_idx': torch.tensor([item['meta']['track_idx'] for item in batch]),
        'bar_idx': torch.tensor([item['meta']['bar_idx'] for item in batch]),
    }
    
    return {
        'bin_level': batched_bin_level,
        'frame_level': batched_frame_level,
        'frame_mask': frame_mask,  # Single mask for all frame-level data
        'bin_position_encoding': bin_position_encoding,
        'beat_position_encoding': beat_position_encoding,
        'meta': meta_data,
        'batch_size': batch_size,
    }

