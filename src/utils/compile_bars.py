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
        'bar_num': row['bar_num'],
    }

def prepare_torch_data(
        song_score_annotations, 
        song_rhythm_signature, 
        song_data_flags, 
        song_raw_rhythm_signature, 
        song_bar_nums, 
        output_folder, 
        fsid, 
        mode,
    ):
        # Convert lists to tensors
    score_annotations_tensor = torch.tensor(song_score_annotations)
    rhythm_signature_tensor = torch.tensor(song_rhythm_signature)
    data_flags_tensor = torch.tensor(song_data_flags)
    bar_nums_tensor = torch.tensor(song_bar_nums)

    # Save tensors in a dictionary
    torch_data = {
        'tokens': score_annotations_tensor,
        'rhythm_signatures': rhythm_signature_tensor,
        'flags': data_flags_tensor,
        'bar_nums': bar_nums_tensor
    }

    torch_data['rhythm_tokens'] = torch.stack([torch.tensor([generate_rhythm_token(x) for x in bar]) for bar in song_raw_rhythm_signature])
    torch_data['inferred_time_feel'] = torch.stack([torch.tensor([generate_inferred_time_feel(x) for x in bar]) for bar in song_raw_rhythm_signature])
    torch_data['source_time_feel'] = torch.tensor(mode == 'original', dtype=torch.int64)
    torch_data['mask'] = torch.stack([torch.tensor([signature_to_mask(x) for x in bar]) for bar in song_raw_rhythm_signature])

    # Save using torch.save for efficient loading
    torch.save(torch_data, os.path.join(output_folder, f'{fsid}.{mode}.pt'))

def compile_filosax(labeled_scores, output_folder):
    for participant in range(1, 6):
        for song in range(1, 49):
            print(f'Processing {participant} {song}...')
            fsid = f'FS{participant}_{song:02d}'
            
            metadata = labeled_scores[labeled_scores['participant'] == participant]
            metadata = metadata[metadata['song'] == song]

            song_score_annotations = []
            song_rhythm_signature = []
            song_data_flags = []
            song_raw_rhythm_signature = []
            song_bar_nums = []

            # double time
            song_score_annotations_dt = []
            song_rhythm_signature_dt = []
            song_data_flags_dt = []
            song_raw_rhythm_signature_dt = []
            song_bar_nums_dt = []

            # double time + 1 beat shift
            song_score_annotations_dts = []
            song_rhythm_signature_dts = []
            song_data_flags_dts = []
            song_raw_rhythm_signature_dts = []
            song_bar_nums_dts = []

            prev_downbeat = 0
            shift = False
            for _, row in metadata.iterrows():
                if row['double_time'] is False:
                    bar_annotation = prepare_annotations(row)
                    song_score_annotations.append(bar_annotation['beatwise_score'])
                    song_rhythm_signature.append(bar_annotation['rhythm_signature'])
                    song_data_flags.append(bar_annotation['flags'])
                    song_raw_rhythm_signature.append(bar_annotation['raw_rhythm_signature'])
                    song_bar_nums.append(bar_annotation['bar_num'])
                elif row['double_time'] is True:
                    if row['beats'][0] < prev_downbeat:
                        shift = True
                    prev_downbeat = row['beats'][0]
                    bar_annotation = prepare_annotations(row)
                    if shift:
                        song_score_annotations_dts.append(bar_annotation['beatwise_score'])
                        song_rhythm_signature_dts.append(bar_annotation['rhythm_signature'])
                        song_data_flags_dts.append(bar_annotation['flags'])
                        song_raw_rhythm_signature_dts.append(bar_annotation['raw_rhythm_signature'])
                        song_bar_nums_dts.append(bar_annotation['bar_num'])
                    else:
                        song_score_annotations_dt.append(bar_annotation['beatwise_score'])
                        song_rhythm_signature_dt.append(bar_annotation['rhythm_signature'])
                        song_data_flags_dt.append(bar_annotation['flags'])
                        song_raw_rhythm_signature_dt.append(bar_annotation['raw_rhythm_signature'])
                        song_bar_nums_dt.append(bar_annotation['bar_num'])

            prepare_torch_data(song_score_annotations, 
                song_rhythm_signature, 
                song_data_flags, 
                song_raw_rhythm_signature, 
                song_bar_nums, 
                output_folder, 
                fsid, 
                'original',
            )

            prepare_torch_data(
                song_score_annotations_dt, 
                song_rhythm_signature_dt, 
                song_data_flags_dt, 
                song_raw_rhythm_signature_dt, 
                song_bar_nums_dt, 
                output_folder, 
                fsid, 
                'double_time',
            )

            prepare_torch_data(
                song_score_annotations_dts, 
                song_rhythm_signature_dts, 
                song_data_flags_dts, 
                song_raw_rhythm_signature_dts, 
                song_bar_nums_dts, 
                output_folder, 
                fsid, 
                'double_time_shifted',
            )

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--labeled_scores', type=str, default='../stages/1_processed_scores/labeled_scores.json')
    parser.add_argument('--output_folder', type=str, default='../stages/1d_compiled_bars')
    args = parser.parse_args()

    print('Loading labeled scores...')
    with open(args.labeled_scores, 'r') as f:
        labeled_scores = pd.DataFrame(json.load(f))
    labeled_scores = labeled_scores.replace({None: np.nan})

    compile_filosax(labeled_scores, args.output_folder)
