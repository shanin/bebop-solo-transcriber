import os
import pandas as pd
import numpy as np
import json
import argparse
import torch
from ..tokenizer.rhythm_tokens import RHYTHM_TOKENS

def rhythm_str_to_int(quaternary_str: str) -> int:
    substitution = {'o': '0', 't': '1', 'r': '2', 'x': '3'}
    quaternary_str = quaternary_str.translate(str.maketrans(substitution))
    return int(quaternary_str, base=4)

def int_to_rhythm_str(k: int) -> str:
    integer = k
    result = 'oooooooooooo'
    if integer != 0:
        substitution = {0: 'o', 1: 't', 2: 'r', 3: 'x'}
        digits = []
        while integer:
            digits.append(int(integer % 4))
            integer //= 4
        str_end = ''.join([substitution[digit] for digit in digits[::-1]])
        result = result[:-len(str_end)] + str_end
    return result

def signature_to_mask(signature: str):
    mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(signature)):
        if signature[i] == 'o':
            mask[i] = 0
        elif signature[i] == 't':
            mask[i] = 1
        elif signature[i] == 'r':
            mask[i] = 2
        elif signature[i] == 'x':
            mask[i] = 3
    return mask

def average_activations(activations: np.ndarray, confidence: np.ndarray):
    normalization = confidence.sum()
    return activations.T @ confidence / normalization

def prepare_beat(activations, confidence, amplitude, flux, bins=12):
    grid = np.linspace(0, len(flux), bins + 1)
    start = grid[:-1]
    end = grid[1:]
    aggregated_activations = np.zeros((bins, activations.shape[1]))
    aggregated_flux = np.zeros(bins)
    aggregated_amplitude = np.zeros(bins)
    aggregated_confidence = np.zeros(bins)
    for bin_idx in range(bins):
        bin_start = int(start[bin_idx])
        bin_end = int(end[bin_idx])
        # Select rows in this bin
        #mask = (time_sec >= bin_start) & (time_sec < bin_end)
        aggregated_activations[bin_idx] = average_activations(activations[bin_start:bin_end], confidence[bin_start:bin_end])
        aggregated_flux[bin_idx] = flux[bin_start:bin_end].mean()
        aggregated_amplitude[bin_idx] = amplitude[bin_start:bin_end].mean()
        aggregated_confidence[bin_idx] = confidence[bin_start:bin_end].mean()
    return {
        'activations': aggregated_activations.tolist(),
        'flux': aggregated_flux.tolist(),
        'amplitude': aggregated_amplitude.tolist(),
        'confidence': aggregated_confidence.tolist()
    }


def prepare_bar(activations, confidence, amplitude, flux, beats, fps=100, bins=12):
    """
    beats: five values in seconds 
    """
    results = []
    for beat_idx in range(4):
        beat_activations = activations[int(beats[beat_idx] * fps):int(beats[beat_idx + 1] * fps)]
        beat_flux = flux[int(beats[beat_idx] * fps):int(beats[beat_idx + 1] * fps)]
        beat_amplitude = amplitude[int(beats[beat_idx] * fps):int(beats[beat_idx + 1] * fps)]
        beat_confidence = confidence[int(beats[beat_idx] * fps):int(beats[beat_idx + 1] * fps)]
        beat_features = prepare_beat(beat_activations, beat_confidence, beat_amplitude, beat_flux, bins)
        results.append(beat_features)
    
    return {
        'activations': np.stack([x['activations'] for x in results], axis=0).tolist(),
        'flux': np.stack([x['flux'] for x in results], axis=0).tolist(), 
        'amplitude': np.stack([x['amplitude'] for x in results], axis=0).tolist(),
        'confidence': np.stack([x['confidence'] for x in results], axis=0).tolist()
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
        'rhythm_tokens': [generate_rhythm_token(row['rhythm_signature'][x]) for x in range(4)],
        'flags': [row['flag1'], row['flag2'], row['flag3'], row['flag4']],
    }

def combine_filosax(labeled_scores, pesto_folder, flux_folder, output_folder):
    for participant in range(1, 6):
        for song in range(1, 49):
            fsid = f'FS{participant}_{song:02d}'

            activations = np.load(os.path.join(pesto_folder, f'{fsid}.activations.npy'))
            confidence = np.load(os.path.join(pesto_folder, f'{fsid}.confidence.npy'))
            amplitude = np.load(os.path.join(pesto_folder, f'{fsid}.amplitude.npy'))
            flux = np.load(os.path.join(flux_folder, f'{fsid}.flux.npy'))[:amplitude.shape[0]]
            

            dt_len = activations.shape[0] - activations.shape[0] % 2
            activations_dt = np.mean(activations[:dt_len].reshape(-1, 2, activations.shape[1]), axis=1)
            flux_dt = np.mean(flux[:dt_len].reshape(-1, 2), axis=1)
            amplitude_dt = np.mean(amplitude[:dt_len].reshape(-1, 2), axis=1)
            confidence_dt = np.mean(confidence[:dt_len].reshape(-1, 2), axis=1)

            metadata = labeled_scores[labeled_scores['participant'] == participant]
            metadata = metadata[metadata['song'] == song]

            song_activations = []
            song_scalar_features = []
            song_score_annotations = []
            song_rhythm_signature = []
            song_data_flags = []
            for _, row in metadata.iterrows():
                if row['double_time'] is False:
                    beats = row['beats']
                    bar_features = prepare_bar(activations, confidence, amplitude, flux, beats, fps=100, bins=12)
                    bar_annotation = prepare_annotations(row)
                    song_activations.append(bar_features['activations'])
                    song_scalar_features.append([bar_features['flux'], bar_features['amplitude'], bar_features['confidence']])
                    song_score_annotations.append(bar_annotation['beatwise_score'])
                    song_rhythm_signature.append(bar_annotation['rhythm_signature'])
                    song_data_flags.append(bar_annotation['flags'])

            # Convert lists to tensors
            activations_tensor = torch.tensor(song_activations)
            scalar_features_tensor = torch.tensor(song_scalar_features) 
            score_annotations_tensor = torch.tensor(song_score_annotations)
            rhythm_signature_tensor = torch.tensor(song_rhythm_signature)
            data_flags_tensor = torch.tensor(song_data_flags)

            # Save tensors in a dictionary
            torch_data = {
                'activations': activations_tensor,
                'scalar_features': scalar_features_tensor,
                'tokens': score_annotations_tensor,
                'rhythm_signatures': rhythm_signature_tensor,
                'flags': data_flags_tensor
            }

            torch_data['scalar_features'] = torch_data['scalar_features'].transpose(1, 2)
            torch_data['rhythm_tokens'] = torch.stack([torch.tensor([generate_rhythm_token(x) for x in bar]) for bar in torch_data['rhythm_signatures']])
            torch_data['inferred_time_feel'] = torch.stack([torch.tensor([generate_inferred_time_feel(x) for x in bar]) for bar in torch_data['rhythm_signatures']])
            torch_data['source_time_feel'] = torch.tensor(True, dtype=torch.int64)
            torch_data['mask'] = torch.stack([torch.tensor([signature_to_mask(x) for x in bar]) for bar in torch_data['rhythm_signatures']])

            # Save using torch.save for efficient loading
            torch.save(torch_data, os.path.join(output_folder, f'{fsid}.original.pt'))

            song_activations = []
            song_scalar_features = []
            song_score_annotations = []
            song_rhythm_signature = []
            song_data_flags = []
            for _, row in metadata.iterrows():
                if row['double_time'] is True:
                    beats = row['beats']
                    bar_features = prepare_bar(activations_dt, confidence_dt, amplitude_dt, flux_dt, beats, fps=100, bins=12)
                    bar_annotation = prepare_annotations(row)
                    song_activations.append(bar_features['activations'])
                    song_scalar_features.append([bar_features['flux'], bar_features['amplitude'], bar_features['confidence']])
                    song_score_annotations.append(bar_annotation['beatwise_score'])
                    song_rhythm_signature.append(bar_annotation['rhythm_signature'])
                    song_data_flags.append(bar_annotation['flags'])

            # Convert lists to tensors
            activations_tensor = torch.tensor(song_activations)
            scalar_features_tensor = torch.tensor(song_scalar_features) 
            score_annotations_tensor = torch.tensor(song_score_annotations)
            rhythm_signature_tensor = torch.tensor(song_rhythm_signature)
            data_flags_tensor = torch.tensor(song_data_flags)

            # Save tensors in a dictionary
            torch_data = {
                'activations': activations_tensor,
                'scalar_features': scalar_features_tensor,
                'tokens': score_annotations_tensor,
                'rhythm_signatures': rhythm_signature_tensor,
                'flags': data_flags_tensor
            }

            torch_data['scalar_features'] = torch_data['scalar_features'].transpose(1, 2)
            torch_data['rhythm_tokens'] = torch.stack([torch.tensor([generate_rhythm_token(x) for x in bar]) for bar in torch_data['rhythm_signatures']])
            torch_data['inferred_time_feel'] = torch.stack([torch.tensor([generate_inferred_time_feel(x) for x in bar]) for bar in torch_data['rhythm_signatures']])
            torch_data['source_time_feel'] = torch.tensor(False, dtype=torch.int64)
            torch_data['mask'] = torch.stack([torch.tensor([signature_to_mask(x) for x in bar]) for bar in torch_data['rhythm_signatures']])

            torch.save(torch_data, os.path.join(output_folder, f'{fsid}.double_time.pt'))

def generate_rhythm_token(signature: str):
    if signature in RHYTHM_TOKENS:
        return RHYTHM_TOKENS[signature]['id']
    elif 'x' in signature:
        return 42 # too fast rhythm
    else:
        return 43 # too rare rhythm
    
def generate_inferred_time_feel(signature: str):
    if signature in RHYTHM_TOKENS:
        if RHYTHM_TOKENS[signature]['feel'] == 'swing':
            return 0
        elif RHYTHM_TOKENS[signature]['feel'] == 'double_time':
            return 1
        else:
            assert False
    elif 'x' in signature:
        return 1
    else:
        return 1 # too rare rhythm 

def combine_omnibook(labeled_scores, pesto_folder, flux_folder, output_folder):
    participant = 'bird'
    songs = labeled_scores[labeled_scores['participant'] == participant]['song'].unique()
    for song in songs:
        if song == 'GRfYc':
            continue

        activations = np.load(os.path.join(pesto_folder, f'OB_{song}.activations.npy'))
        confidence = np.load(os.path.join(pesto_folder, f'OB_{song}.confidence.npy'))
        amplitude = np.load(os.path.join(pesto_folder, f'OB_{song}.amplitude.npy'))
        flux = np.load(os.path.join(flux_folder, f'OB_{song}.flux.npy'))[:amplitude.shape[0]]
        
        metadata = labeled_scores[labeled_scores['song'] == song]

        song_activations = []
        song_scalar_features = []
        song_score_annotations = []
        song_rhythm_signature = []
        song_data_flags = []
        for _, row in metadata.iterrows():
            beats = row['beats']
            bar_features = prepare_bar(activations, confidence, amplitude, flux, beats, fps=100, bins=12)
            bar_annotation = prepare_annotations(row)
            song_activations.append(bar_features['activations'])
            song_scalar_features.append([bar_features['flux'], bar_features['amplitude'], bar_features['confidence']])
            song_score_annotations.append(bar_annotation['beatwise_score'])
            song_rhythm_signature.append(bar_annotation['rhythm_signature'])
            song_data_flags.append(bar_annotation['flags'])

        activations_tensor = torch.tensor(song_activations)
        scalar_features_tensor = torch.tensor(song_scalar_features) 
        score_annotations_tensor = torch.tensor(song_score_annotations)
        rhythm_signature_tensor = torch.tensor(song_rhythm_signature)
        data_flags_tensor = torch.tensor(song_data_flags)

        torch_data = {
            'activations': activations_tensor,
            'scalar_features': scalar_features_tensor,
            'tokens': score_annotations_tensor,
            'rhythm_signatures': rhythm_signature_tensor,
            'flags': data_flags_tensor
        }
        torch_data['scalar_features'] = torch_data['scalar_features'].transpose(1, 2)
        torch_data['rhythm_tokens'] = torch.stack([torch.tensor([generate_rhythm_token(x) for x in bar]) for bar in torch_data['rhythm_signatures']])
        torch_data['inferred_time_feel'] = torch.stack([torch.tensor([generate_inferred_time_feel(x) for x in bar]) for bar in torch_data['rhythm_signatures']])
        torch_data['source_time_feel'] = torch.tensor(True, dtype=torch.int64)
        torch_data['mask'] = torch.stack([torch.tensor([signature_to_mask(x) for x in bar]) for bar in torch_data['rhythm_signatures']])

        torch.save(torch_data, os.path.join(output_folder, f'OB_{song}.original.pt'))

               
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--labeled_scores', type=str, default='../stages/1_processed_scores/labeled_scores.json')
    parser.add_argument('--pesto_folder', type=str, default='../test')
    parser.add_argument('--flux_folder', type=str, default='../test')
    parser.add_argument('--output_folder', type=str, default='../stages/1b_combined_data')
    parser.add_argument('--mode', type=str, default='filosax')
    args = parser.parse_args()

    with open(args.labeled_scores, 'r') as f:
        labeled_scores = pd.DataFrame(json.load(f))

    if args.mode == 'filosax':
        combine_filosax(labeled_scores, args.pesto_folder, args.flux_folder, args.output_folder)
    elif args.mode == 'omnibook':
        combine_omnibook(labeled_scores, args.pesto_folder, args.flux_folder, args.output_folder)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')