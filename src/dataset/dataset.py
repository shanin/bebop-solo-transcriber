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

class SoloDataset(Dataset):
    def __init__(self, data_dir=None, source: str = 'original', split: str = 'all'):
        self.data_dir = data_dir
        self.rhythm_tokens = RHYTHM_TOKENS
        self.source = source
        self.all_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.json')]
        self.split = split
        if self.split != 'all':
            self.prepare_splits()
        self.prepare_file_list()
        self.load_bars()

    def load_bars(self):
        self.data = []
        
        # Load all data into memory
        print(f"Loading {len(self.files)} files from {self.data_dir}, mode: {self.source}, split: {self.split}")
        for file in tqdm(self.files):
            file_path = os.path.join(self.data_dir, file)
            with open(file_path, 'r') as f:
                bars = json.load(f)
            self.data.extend(bars)

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
        
    def generate_rhythm_token(self, signature):
        if signature in self.rhythm_tokens:
            return self.rhythm_tokens[signature]['id']
        elif 'x' in signature:
            return 42 # too fast rhythm
        else:
            return 43 # too rare rhythm
        
    def generate_inferred_time_feel(self, signature):
        if signature in self.rhythm_tokens:
            if self.rhythm_tokens[signature]['feel'] == 'swing':
                return 0
            elif self.rhythm_tokens[signature]['feel'] == 'double_time':
                return 1
            else:
                assert False
        elif 'x' in signature:
            return 1
        else:
            return 1 # too rare rhythm 

    def generate_mask(self, signature):
        mask = torch.zeros(12, dtype=torch.int64)
        for index, x in enumerate(signature):
            if x == 'o':
                mask[index] = 0
            elif x == 'r':
                mask[index] = 1
            elif x == 't':
                mask[index] = 2
            elif x == 'x':
                mask[index] = 3
            else:
                assert False
        return mask

    def __len__(self):
        return len(self.data)
    
    def fix_tokens(self, tokens: torch.Tensor):
        tokens[tokens == 129] = 128 # tie token
        tokens[tokens == 130] = 129 # rest token
        return tokens

    def __getitem__(self, idx):
        bar = self.data[idx]
        flux = torch.tensor(bar['features']['flux']).view(48)
        confidence = torch.tensor(bar['features']['confidence']).view(48)
        amplitude = torch.tensor(bar['features']['amplitude']).view(48)
        activations = torch.tensor(bar['features']['activations']).view(48, 384)
        features = torch.stack([flux, confidence, amplitude], dim = -1)

        tokens = torch.concatenate([
            torch.tensor(bar['annotation'][0]['beatwise_score']),
            torch.tensor(bar['annotation'][1]['beatwise_score']),
            torch.tensor(bar['annotation'][2]['beatwise_score']),
            torch.tensor(bar['annotation'][3]['beatwise_score']),
        ])
        tokens = self.fix_tokens(tokens)

        mask = torch.concatenate([
            self.generate_mask(bar['annotation'][0]['rhythm_signature']),
            self.generate_mask(bar['annotation'][1]['rhythm_signature']),
            self.generate_mask(bar['annotation'][2]['rhythm_signature']),
            self.generate_mask(bar['annotation'][3]['rhythm_signature']),
        ])

        rhythm_tokens = torch.tensor([
            self.generate_rhythm_token(bar['annotation'][0]['rhythm_signature']),
            self.generate_rhythm_token(bar['annotation'][1]['rhythm_signature']),
            self.generate_rhythm_token(bar['annotation'][2]['rhythm_signature']),
            self.generate_rhythm_token(bar['annotation'][3]['rhythm_signature']),
        ])

        inferred_time_feel = torch.tensor([
            self.generate_inferred_time_feel(bar['annotation'][0]['rhythm_signature']),
            self.generate_inferred_time_feel(bar['annotation'][1]['rhythm_signature']),
            self.generate_inferred_time_feel(bar['annotation'][2]['rhythm_signature']),
            self.generate_inferred_time_feel(bar['annotation'][3]['rhythm_signature']),
        ])

        source_time_feel = torch.tensor(self.source != 'original', dtype=torch.int64)


        return {
            'features': features,
            'activations': activations,
            'tokens': tokens,
            'mask': mask,
            'rhythm_tokens': rhythm_tokens,
            'inferred_time_feel': inferred_time_feel,
            'source_time_feel': source_time_feel,
        }
    

class OmnibookDataset(SoloDataset):
    def prepare_splits(self):
        self.test_files = [
            'OB_1p64c.original.json', 'OB_S5VYc.original.json', 'OB_wkTyc.original.json',
        ]
        self.val_files = [
            'OB_Nqn4c.original.json', 'OB_6Cbwc.original.json'
        ]
        self.train_files = [f for f in self.all_files if f not in self.test_files and f not in self.val_files]


class FilosaxDataset(SoloDataset):
    def prepare_splits(self):
        self.test_files = [f'FS{i}_46.{self.source}.json' for i in range(1, 6)] + \
                    [f'FS{i}_47.{self.source}.json' for i in range(1, 6)] + \
                    [f'FS{i}_48.{self.source}.json' for i in range(1, 6)]
        self.val_files = [f'FS{i}_45.{self.source}.json' for i in range(1, 6)]
        self.train_files = [f for f in self.all_files if f not in self.test_files and f not in self.val_files and f.endswith(f'.{self.source}.json')]

class ConsecutiveBarSampler(Sampler):
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
    
def collate_fn(batch, num_consecutive_bars, random_transposition: bool = False, mode: str = 'tenor'):
    # Calculate actual batch size from the input
    actual_batch_size = len(batch) // num_consecutive_bars

    # Stack features
    features = torch.stack([item['features'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 48, 3)
    
    # Stack activations
    activations = torch.stack([item['activations'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 48, 384)
    
    # Stack tokens
    tokens = torch.stack([item['tokens'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 48)
    mask = torch.stack([item['mask'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 48)
    inferred_time_feel = torch.stack([item['inferred_time_feel'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 4)
    source_time_feel = torch.stack([item['source_time_feel'] for item in batch]).view(actual_batch_size, num_consecutive_bars, 1)
    
    # Convert rhythm tokens to tensors and stack
    rhythm_tokens = torch.stack([torch.tensor(item['rhythm_tokens']) for item in batch]).view(actual_batch_size, num_consecutive_bars, 4)
    
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

def dataloader_generator(dataset,  
                        num_consecutive_bars: int,
                        random_transposition: bool = False,
                        inference: bool = False,
                        batch_size: int = 32,
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
            num_workers=0,
            collate_fn=collate_local_fn,
        )