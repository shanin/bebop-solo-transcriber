import pickle
import pandas as pd
import argparse
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset', type=str, default='filosax')
    args = parser.parse_args()

    scores = pickle.load(open(args.input_file, 'rb'))
    if args.dataset == 'filosax':
        for participant in range(1, 6):
            for song in range(1, 49):
                local_scores = scores[scores['participant'] == participant]
                local_scores = local_scores[local_scores['song'] == song]
                features = np.load(os.path.join(args.input_dir, f'FS{participant}_{song:02d}.onsets.npy'))
                bar_index = [-1] * len(features)
                beat_index = [-1] * len(features)
                frame_phase = [-1] * len(features)
                frame_centers = np.arange(len(features)) / 100 + 0.005
                fourier = np.zeros((len(features), 12))

                for i, bar in local_scores.iterrows():
                    bar_start = bar.syncpoint
                    if i == len(scores) - 1:
                        bar_end = bar_start + 4 * delta
                    else:
                        next_bar = scores.iloc[i+1]
                        bar_end = next_bar.syncpoint
                    delta = (bar_end - bar_start) / 4
                    beats = [bar_start, bar_start + delta, bar_start + 2 * delta, bar_start + 3 * delta, bar_end]
                    mask = (frame_centers > beats[0]) & (frame_centers < beats[-1])
                    for idx in np.where(mask)[0]:
                        bar_index[idx] = bar.bar_num
                    for i in range(4):
                        mask = (frame_centers > beats[i]) & (frame_centers < beats[i+1])
                        for idx in np.where(mask)[0]:
                            beat_index[idx] = i
                            phase = (frame_centers[idx] - beats[i]) / delta
                            frame_phase[idx] = phase
                            for j in range(6):
                                fourier[idx, 2*j] = np.sin(2 * np.pi * phase * (j + 1))
                                fourier[idx, 2*j+1] = np.cos(2 * np.pi * phase * (j + 1))
                
                # Convert 1D arrays to 2D for concatenation
                bar_index = np.array(bar_index).reshape(-1, 1)
                beat_index = np.array(beat_index).reshape(-1, 1)
                frame_phase = np.array(frame_phase).reshape(-1, 1)
                
                posenc = np.concatenate([bar_index, beat_index, frame_phase, fourier], axis=-1)
                np.save(os.path.join(args.output_dir, f'FS{participant}_{song:02d}.posenc.npy'), posenc)

