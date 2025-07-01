from torch.utils.data import Dataset
import torch
from src.tokenizer.rhythm_tokens import RHYTHM_TOKENS
import os
import json

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
        for file in self.files:
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