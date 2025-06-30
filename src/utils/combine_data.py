import os
import pandas as pd
import numpy as np
import json
import argparse

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
    return results

def prepare_annotations(row, source):
    result = []
    for x in [0,1,2,3]:
        result.append({
            'beatwise_score': row['beatwise_score'][x],
            'rhythm_signature': row['rhythm_signature'][x],
            'flags': [row['flag1'], row['flag2'], row['flag3'], row['flag4']],
        })
    return result

def combine_filosax(labeled_scores, pesto_folder, flux_folder, output_folder):
    for participant in range(1, 6):
        for song in range(1, 49):
                fsid = f'FS{participant}_{song:02d}'

                activations = np.load(os.path.join(pesto_folder, f'{fsid}.activations.npy'))
                confidence = np.load(os.path.join(pesto_folder, f'{fsid}.confidence.npy'))
                amplitude = np.load(os.path.join(pesto_folder, f'{fsid}.amplitude.npy'))
                flux = np.load(os.path.join(flux_folder, f'{fsid}.flux.npy'))[:amplitude.shape[0]]
                
                activations_dt = np.mean(activations.reshape(-1, 2, activations.shape[1]), axis=1)
                flux_dt = np.mean(flux.reshape(-1, 2), axis=1)
                amplitude_dt = np.mean(amplitude.reshape(-1, 2), axis=1)
                confidence_dt = np.mean(confidence.reshape(-1, 2), axis=1)

                metadata = labeled_scores[labeled_scores['participant'] == participant]
                metadata = metadata[metadata['song'] == song]

                features = []
                for _, row in metadata.iterrows():
                    if row['double_time'] is False:
                        beats = row['beats']
                        bar_features = prepare_bar(activations, confidence, amplitude, flux, beats, fps=100, bins=12)
                        features.append({'features':bar_features, 'annotation': prepare_annotations(row, source='original')})

                with open(os.path.join(output_folder, f'{fsid}.original.json'), 'w') as f:
                    json.dump(features, f)

                features_dt = []
                for _, row in metadata.iterrows():
                    if row['double_time'] is True:
                        beats = row['beats']
                        bar_features = prepare_bar(activations_dt, confidence_dt, amplitude_dt, flux_dt, beats, fps=100, bins=12)
                        features_dt.append({'features':bar_features, 'annotation': prepare_annotations(row, source='double_time')})

                with open(os.path.join(output_folder, f'{fsid}.double_time.json'), 'w') as f:
                    json.dump(features_dt, f)

def combine_omnibook(labeled_scores, pesto_folder, flux_folder, output_folder):
    participant = 'bird'
    songs = labeled_scores[labeled_scores['participant'] == participant]['song'].unique()
    for song in songs:

        activations = np.load(os.path.join(pesto_folder, f'OB_{song}.activations.npy'))
        confidence = np.load(os.path.join(pesto_folder, f'OB_{song}.confidence.npy'))
        amplitude = np.load(os.path.join(pesto_folder, f'OB_{song}.amplitude.npy'))
        flux = np.load(os.path.join(flux_folder, f'OB_{song}.flux.npy'))[:amplitude.shape[0]]
        
        metadata = labeled_scores[labeled_scores['song'] == song]

        features = []
        for _, row in metadata.iterrows():
            beats = row['beats']
            bar_features = prepare_bar(activations, confidence, amplitude, flux, beats, fps=100, bins=12)
            features.append({'features':bar_features, 'annotation': prepare_annotations(row, source='original')})

        with open(os.path.join(output_folder, f'OB_{song}.original.json'), 'w') as f:
            json.dump(features, f)

               
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