import pandas as pd
import json
import numpy as np
import os
import pickle
import argparse


def fill_beats(metadata):
    # fill beats
    metadata.dropna(inplace=True, subset=['syncpoint'])
    beats = []
    for index in range(len(metadata) - 1):
        start = metadata.iloc[index]['syncpoint']
        end = metadata.iloc[index + 1]['syncpoint']
        beat = - (start - end) / 4
        local_beats = [
            start,
            start + beat,
            start + 2 * beat,
            start + 3 * beat,
            end
        ]
        beats.append(local_beats)
    last_beat = metadata.iloc[-1]['syncpoint']
    beats.append([last_beat, last_beat + beat, last_beat + 2 * beat, last_beat + 3 * beat, last_beat + 4 * beat])
    metadata['beats'] = beats
    return metadata

def downsample(tokens, bar_num=None):
    new_tokens = []
    for i in range(len(tokens) // 2):
        if tokens[2*i] == tokens[2*i+1]:
            new_tokens.append(int(tokens[2*i]))
        elif tokens[2*i] < 129 and tokens[2*i+1] == 129:
            new_tokens.append(int(tokens[2*i]))
        #elif tokens[2*i] == 129 and tokens[2*i+1] == 130:
        #    print(f'WARNING: Unexpected token pair: 129 130 at bar {bar_num}')
        #    new_tokens.append(tokens[2*i])
        else:
            print(f"Unexpected token pair: {tokens[2*i]} {tokens[2*i+1]} at bar {bar_num}")
            new_tokens.append(131)
    return new_tokens

def rhythm_str(tokens):
    result = []
    for token in tokens:
        if token == 130:
            result.append('r')
        elif token == 129:
            result.append('t') 
        elif 0 <= token <= 128:
            result.append('o')
        elif token == 131:
            result.append('x')
        else:
            assert False, f"Unexpected token value: {token}"
    return ''.join(result)

def normal(metadata):
    mode = 'normal'
    new_rows = []
    for index, row in metadata.iterrows():
        if not isinstance(row['beatwise_score'], list):
            row_score = [[130] * 12] * 4
            #assert row['flag1'] != row['flag1'] and row['flag2'] != row['flag2'], f"Unexpected flag values at bar {index}: {row['flag1']} {row['flag2']}"
        else:
            row_score = [downsample(x, index) for x in row['beatwise_score']]
        rhythm_signature = [rhythm_str(x) for x in row_score]
        new_rows.append({
            'participant': row['participant'],
            'song': row['song'],
            'bar_num': row['bar_num'],
            'beatwise_score': row_score,
            'rhythm_signature': rhythm_signature,
            'syncpoint': row['syncpoint'],
            'measure_offset': row['measure_offset'],
            'beats': row['beats'],
            'flag1': row['flag1'],
            'flag2': row['flag2'],
            'flag3': np.nan,
            'flag4': np.nan,
            'double_time': False
        })
    normal_scores = pd.DataFrame(new_rows)
    return normal_scores

def double_time(metadata):
    mode = 'double_time'
    new_rows = []
    for index in range(len(metadata) // 2):
        row_1 = metadata.iloc[2*index]
        row_2 = metadata.iloc[2*index+1]
        if not isinstance(row_1['beatwise_score'], list):
            row_1_score = [[130] * 12] * 4
            assert row_1['flag1'] != row_1['flag1'] and row_1['flag2'] != row_1['flag2'], f"Unexpected flag values at bar {index}: {row_1['flag1']} {row_1['flag2']}"
        else:
            row_1_score = [downsample(x, index) for x in row_1['beatwise_score']]
        if not isinstance(row_2['beatwise_score'], list):
            row_2_score = [[130] * 12] * 4
            assert row_2['flag1'] != row_2['flag1'] and row_2['flag2'] != row_2['flag2'], f"Unexpected flag values at bar {index}: {row_2['flag1']} {row_2['flag2']}"
        else:
            row_2_score = [downsample(x, index) for x in row_2['beatwise_score']]
        merged = [row_1_score[0] + row_1_score[1], row_1_score[2] + row_1_score[3], row_2_score[0] + row_2_score[1], row_2_score[2] + row_2_score[3]]
        new_score = [downsample(x, index) for x in merged]
        rhythm_signature = [rhythm_str(x) for x in new_score]
            
        new_beats = [row_1['beats'][0] / 2, row_1['beats'][2] / 2, row_2['beats'][0] / 2, row_2['beats'][2] / 2]

        new_rows.append({
            'participant': row_1['participant'],
            'song': row_1['song'],
            'bar_num': row_1['bar_num'] / 2,
            'beatwise_score': new_score,
            'rhythm_signature': rhythm_signature,
            'syncpoint': row_1['syncpoint'] / 2,
            'measure_offset': row_1['measure_offset'] / 2,
            'beats': new_beats,
            'flag1': row_1['flag1'],
            'flag2': row_1['flag2'],
            'flag3': row_2['flag1'],
            'flag4': row_2['flag2'],
            'double_time': True
        })
    double_time_scores = pd.DataFrame(new_rows)
    return double_time_scores

def double_time_shifted(metadata):
    new_rows = []
    for index in range(len(metadata) // 2 - 2):
        row_1 = metadata.iloc[2*index]
        row_2 = metadata.iloc[2*index+1]
        row_3 = metadata.iloc[2*index+2]

        if not isinstance(row_1['beatwise_score'], list):
            row_1_score = [[130] * 12] * 4
            assert row_1['flag1'] != row_1['flag1'] and row_1['flag2'] != row_1['flag2'], f"Unexpected flag values at bar {index}: {row_1['flag1']} {row_1['flag2']}"
        else:
            row_1_score = [downsample(x, index) for x in row_1['beatwise_score']]
        if not isinstance(row_2['beatwise_score'], list):
            row_2_score = [[130] * 12] * 4
            assert row_2['flag1'] != row_2['flag1'] and row_2['flag2'] != row_2['flag2'], f"Unexpected flag values at bar {index}: {row_2['flag1']} {row_2['flag2']}"
        else:
            row_2_score = [downsample(x, index) for x in row_2['beatwise_score']]
        if not isinstance(row_3['beatwise_score'], list):
            row_3_score = [[130] * 12] * 4
            assert row_3['flag1'] != row_3['flag1'] and row_3['flag2'] != row_3['flag2'], f"Unexpected flag values at bar {index}: {row_3['flag1']} {row_3['flag2']}"
        else:
            row_3_score = [downsample(x, index) for x in row_3['beatwise_score']]

        merged = [row_1_score[1] + row_1_score[2], row_1_score[3] + row_2_score[0], row_2_score[1] + row_2_score[2], row_2_score[3] + row_3_score[0]]
        new_score = [downsample(x, index) for x in merged]
        rhythm_signature = [rhythm_str(x) for x in new_score]
            
        new_beats = [row_1['beats'][1] / 2, row_1['beats'][3] / 2, row_2['beats'][1] / 2, row_2['beats'][3] / 2]

        new_rows.append({
            'participant': row_1['participant'],
            'song': row_1['song'],
            'bar_num': row_1['bar_num'] / 2,
            'beatwise_score': new_score,
            'rhythm_signature': rhythm_signature,
            'syncpoint': new_beats[0],
            'measure_offset': row_1['measure_offset'] / 2,
            'beats': new_beats,
            'flag1': row_1['flag1'],
            'flag2': row_1['flag2'],
            'flag3': row_2['flag1'],
            'flag4': row_2['flag2'],
            'double_time': True
        })
    double_time_shifted_scores = pd.DataFrame(new_rows)
    return double_time_shifted_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='../stages/1_processed_scores/combined_scores.pkl')
    parser.add_argument('--output_dir', type=str, default='../stages/1_processed_scores')
    args = parser.parse_args()

    scores = pickle.load(open(args.input_file, 'rb'))

    normal_scores = []
    double_time_scores = []
    double_time_shifted_scores = []
    for participant in range(1, 6):
        for song in range(1, 49):
            metadata = scores[scores['participant'] == participant]
            metadata = metadata[metadata['song'] == song]
            metadata = fill_beats(metadata)
            normal_scores.append(normal(metadata))
            double_time_scores.append(double_time(metadata))
            double_time_shifted_scores.append(double_time_shifted(metadata))
    normal_scores = pd.concat(normal_scores)
    double_time_scores = pd.concat(double_time_scores)
    double_time_shifted_scores = pd.concat(double_time_shifted_scores)
    all_filosax_scores = pd.concat([normal_scores, double_time_scores, double_time_shifted_scores])

    normal_scores = []
    double_time_scores = []
    double_time_shifted_scores = []
    participant = 'bird'
    filtered_scores = scores[scores['song'] != 'gRfYc']
    songs = filtered_scores[filtered_scores['participant'] == participant]['song'].unique()
    assert 'gRfYc' not in songs
    for song in songs:
        print(song)
        metadata = scores[scores['participant'] == participant]
        metadata = metadata[metadata['song'] == song]
        metadata = fill_beats(metadata)
        normal_scores.append(normal(metadata))
    omnibook_scores = pd.concat(normal_scores)

    all_scores = pd.concat([omnibook_scores, all_filosax_scores])
    all_scores.to_pickle(os.path.join(args.output_dir, 'labeled_scores.pkl'))