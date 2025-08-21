import os
import pandas as pd
import numpy as np
import json
import argparse
import torch

from src.tokenizer.rhythm_utils import (
    rhythm_str_to_int, 
    signature_to_mask,
    generate_inferred_time_feel,
    generate_rhythm_token
)


def prepare_bar(onsets, offsets, frames, posenc, bar_num=0):

    bar_index = posenc[:, 0]
    beat_index = posenc[:, 1]

    bar_onsets = []
    bar_offsets = []
    bar_frames = []
    bar_posenc = []

    for beat_idx in range(4):

        beat_onsets = onsets[(beat_index == beat_idx) & (bar_index == bar_num)]
        beat_offsets = offsets[(beat_index == beat_idx) & (bar_index == bar_num)]
        beat_frames = frames[(beat_index == beat_idx) & (bar_index == bar_num)]
        beat_posenc = posenc[(beat_index == beat_idx) & (bar_index == bar_num)]

        bar_onsets.append(beat_onsets.tolist())
        bar_offsets.append(beat_offsets.tolist())
        bar_frames.append(beat_frames.tolist())
        bar_posenc.append(beat_posenc.tolist())
    
    return {
        'onsets': bar_onsets,
        'offsets': bar_offsets,
        'frames': bar_frames,
        'posenc': bar_posenc,
    }


def fix_score(score):
    score = np.array(score)
    score[score == 129] = 128 # tie token
    score[score == 130] = 129 # rest token
    return score


def prepare_annotations(row):
    return {
        'beatwise_score': [fix_score(row['beatwise_score'][x]) for x in range(4)],
        'rhythm_signature': [rhythm_str_to_int(row['rhythm_signature'][x]) for x in range(4)],
        'raw_rhythm_signature': [row['rhythm_signature'][x] for x in range(4)],
        'rhythm_tokens': [generate_rhythm_token(row['rhythm_signature'][x]) for x in range(4)],
        'flags': [row['flag1'], row['flag2'], row['flag3'], row['flag4']],
    }

def combine_filosax(labeled_scores, crnn_folder, posenc_folder, output_folder):
    for participant in range(1, 6):
        for song in range(1, 49):
            print(f'Processing {participant} {song}...')
            fsid = f'FS{participant}_{song:02d}'

            onsets = np.load(os.path.join(crnn_folder, f'{fsid}.onsets.npy'))
            offsets = np.load(os.path.join(crnn_folder, f'{fsid}.offsets.npy'))
            frames = np.load(os.path.join(crnn_folder, f'{fsid}.frames.npy'))
            posenc = np.load(os.path.join(posenc_folder, f'{fsid}.posenc.npy'))
            
            metadata = labeled_scores[labeled_scores['participant'] == participant]
            metadata = metadata[metadata['song'] == song]

            song_onsets = []
            song_offsets = []
            song_frames = []
            song_posenc = []
            song_score_annotations = []
            song_rhythm_signature = []
            song_data_flags = []
            song_raw_rhythm_signature = []
            for _, row in metadata.iterrows():
                if row['double_time'] is False:
                    bar_features = prepare_bar(onsets, offsets, frames, posenc, bar_num=row['bar_num'])
                    bar_annotation = prepare_annotations(row)
                    song_onsets.append(bar_features['onsets'])
                    song_offsets.append(bar_features['offsets'])
                    song_frames.append(bar_features['frames'])
                    song_posenc.append(bar_features['posenc'])
                    song_score_annotations.append(bar_annotation['beatwise_score'])
                    song_rhythm_signature.append(bar_annotation['rhythm_signature'])
                    song_data_flags.append(bar_annotation['flags'])
                    song_raw_rhythm_signature.append(bar_annotation['raw_rhythm_signature'])

            # Convert lists to tensors
            onsets_tensor = torch.tensor(song_onsets)
            offsets_tensor = torch.tensor(song_offsets)
            frames_tensor = torch.tensor(song_frames)
            posenc_tensor = torch.tensor(song_posenc)
            score_annotations_tensor = torch.tensor(song_score_annotations)
            rhythm_signature_tensor = torch.tensor(song_rhythm_signature)
            data_flags_tensor = torch.tensor(song_data_flags)

            # Save tensors in a dictionary
            torch_data = {
                'onsets': onsets_tensor,
                'offsets': offsets_tensor,
                'frames': frames_tensor,
                'posenc': posenc_tensor,
                'tokens': score_annotations_tensor,
                'rhythm_signatures': rhythm_signature_tensor,
                'flags': data_flags_tensor
            }

            torch_data['onsets'] = torch_data['onsets'].transpose(1, 2)
            torch_data['offsets'] = torch_data['offsets'].transpose(1, 2)
            torch_data['frames'] = torch_data['frames'].transpose(1, 2)
            torch_data['posenc'] = torch_data['posenc'].transpose(1, 2)
            torch_data['rhythm_tokens'] = torch.stack([torch.tensor([generate_rhythm_token(x) for x in bar]) for bar in song_raw_rhythm_signature])
            torch_data['inferred_time_feel'] = torch.stack([torch.tensor([generate_inferred_time_feel(x) for x in bar]) for bar in song_raw_rhythm_signature])
            torch_data['source_time_feel'] = torch.tensor(True, dtype=torch.int64)
            torch_data['mask'] = torch.stack([torch.tensor([signature_to_mask(x) for x in bar]) for bar in song_raw_rhythm_signature])

            # Save using torch.save for efficient loading
            torch.save(torch_data, os.path.join(output_folder, f'{fsid}.original.pt'))

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--labeled_scores', type=str, default='../stages/1_processed_scores/labeled_scores.json')
    parser.add_argument('--crnn_folder', type=str, default='../test')
    parser.add_argument('--posenc_folder', type=str, default='../test')
    parser.add_argument('--output_folder', type=str, default='../stages/1c_combined_data')
    args = parser.parse_args()

    print('Loading labeled scores...')
    with open(args.labeled_scores, 'r') as f:
        labeled_scores = pd.DataFrame(json.load(f))
    labeled_scores = labeled_scores.replace({None: np.nan})

    combine_filosax(labeled_scores, args.crnn_folder, args.posenc_folder, args.output_folder)
