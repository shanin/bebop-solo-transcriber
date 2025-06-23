import pandas as pd
import numpy as np
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['filosax', 'omnibook', 'both'], default='both')
    parser.add_argument('--filosax_path', type=str, required=True)
    parser.add_argument('--omnibook_path', type=str, required=True)
    args = parser.parse_args()

    if args.dataset in ['filosax', 'both']:
        lines = []
        # Filosax
        for participant in range(1, 6):
            for song in range(1, 49):
                id_ = f'FS{participant}_{song:02d}'
                lines.append({
                    'example_id': id_,
                    'participant': participant,
                    'song': song,
                    'dataset': 'filosax',
                    'clean_solo': f'{args.filosax_path}/Participant {participant}/{song:02d}/Sax.wav',
                    'backing_pd': f'{args.filosax_path}/Backing/{song:02d}/Piano_Drums.wav',
                    'backing_bd': f'{args.filosax_path}/Backing/{song:02d}/Bass_Drums.wav',
                    'mix_path': np.nan,
                })
        df = pd.DataFrame(lines)
        df.to_csv('stages/1_processed_scores/index_filosax.csv', index=False)

    if args.dataset in ['omnibook', 'both']:
        lines = []
        for song_file in os.listdir(f'{args.omnibook_path}/audio_stems'):
            if song_file.endswith('.wav'):
                song_id = song_file.split('.wav')[0]
                lines.append({
                    'example_id': f'OB_{song_id}',
                    'participant': 'bird',
                    'song': song_id,
                    'dataset': 'omnibook',
                    'clean_solo': f'{args.omnibook_path}/audio_stems/{song_file}',
                    'backing_pd': np.nan,
                    'backing_bd': np.nan,
                    'mix_path': f'{args.omnibook_path}/audio_untuned_mixes/{song_file.replace(".wav", ".mp3")}',
                })

        df = pd.DataFrame(lines)
        df.to_csv('stages/1_processed_scores/index_omnibook.csv', index=False)