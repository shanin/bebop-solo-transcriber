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
    def __init__(self, data_dir_x=None, data_dir_y=None,  split: str = 'all'):
        self.data_dir_x = data_dir_x
        self.data_dir_y = data_dir_y

        self.all_x_files = [f for f in sorted(os.listdir(data_dir_x)) if f.endswith(f'.melspec.npy')]
        self.all_y_files = [f for f in sorted(os.listdir(data_dir_y)) if f.endswith(f'.frame_labels.npy')]

        self.split = split
        self.instrument = 'none'
        if self.split != 'all':
            self.prepare_splits()
        self.prepare_file_list()

    def prepare_file_list(self):
        if self.split == 'train':
            self.files_x = self.train_files_x
            self.files_y = self.train_files_y
        elif self.split == 'val':
            self.files_x = self.val_files_x
            self.files_y = self.val_files_y
        elif self.split == 'test':
            self.files_x = self.test_files_x
            self.files_y = self.test_files_y
        elif self.split == 'xr_test':
            self.files_x = self.xr_test_files_x
            self.files_y = self.xr_test_files_y
        elif self.split == 'all':
            self.files_x = self.all_x_files 
            self.files_y = self.all_y_files
        else:
            assert False, f"Invalid split: {self.split}"

    def __len__(self):
        return len(self.files_x)

    def __getitem__(self, idx):
        file_path_x = os.path.join(self.data_dir_x, self.files_x[idx])
        file_path_y = os.path.join(self.data_dir_y, self.files_y[idx])
        x = np.load(file_path_x, allow_pickle=True)
        y = np.load(file_path_y, allow_pickle=True).item()
        assert file_path_x.split('.')[-1] == file_path_y.split('.')[-1], f"File names do not match: {file_path_x} and {file_path_y}"
        length = min(x.shape[0], y['onset'].shape[0], y['offset'].shape[0], y['frames'].shape[0])
        x = x[:length]
        y['onset'] = y['onset'][:length]
        y['offset'] = y['offset'][:length]
        y['frames'] = y['frames'][:length]
        return {
            'mel_spec': x,
            'onset': y['onset'],
            'offset': y['offset'],
            'frames': y['frames']
        }


class FrontendSegmentDataset(Dataset):
    def __init__(self, dataset, num_consecutive_frames: int, use_cache: bool = True):
        """
        Args:
            dataset: FrontendTrackDataset object
            use_cache: bool
        """
        self.dataset = dataset
        self.num_consecutive_frames = num_consecutive_frames
        self.index = []
        self.cache = {}
        self.use_cache = use_cache
        for track_idx, track in enumerate(self.dataset):
            num_frames = track['mel_spec'].shape[0]
            for i in range(0, num_frames - self.num_consecutive_frames + 1, self.num_consecutive_frames):
                self.index.append((track_idx, i))
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
        segment = {
            'x': {
                'mel_spec': track['mel_spec'][frame_idx:frame_idx + self.num_consecutive_frames],
            },
            'y': {
                'onset': track['onset'][frame_idx:frame_idx + self.num_consecutive_frames],
                'offset': track['offset'][frame_idx:frame_idx + self.num_consecutive_frames],
                'frames': track['frames'][frame_idx:frame_idx + self.num_consecutive_frames],
            },
        }
        return segment