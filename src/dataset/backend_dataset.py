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
        onsets_data = torch.from_numpy(np.load(onsets_file_path))
        data['onsets'] = onsets_data

        offsets_file_path = os.path.join(self.crnn_dir, self.files[idx].replace(f'.{self.source}.pt', '.offsets.npy'))
        offsets_data = torch.from_numpy(np.load(offsets_file_path))
        data['offsets'] = offsets_data

        frames_file_path = os.path.join(self.crnn_dir, self.files[idx].replace(f'.{self.source}.pt', '.frames.npy'))
        frames_data = torch.from_numpy(np.load(frames_file_path))
        data['frames'] = frames_data

        posenc_file_path = os.path.join(self.posenc_dir, self.files[idx].replace(f'.{self.source}.pt', '.posenc.npy'))
        posenc_data = torch.from_numpy(np.load(posenc_file_path))
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
