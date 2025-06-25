import pandas as pd
import numpy as np
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['filosax', 'uvr_filosax', 'omnibook', 'all'], default='all')
    parser.add_argument('--filosax_path', type=str, required=True)
    parser.add_argument('--uvr_filosax_path', type=str, required=True)
    parser.add_argument('--omnibook_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    if args.dataset in ['filosax', 'all']:
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
        df.to_csv(f'{args.output_dir}/index_filosax.csv', index=False)

    if args.dataset in ['uvr_filosax', 'all']:
        lines = []
        suffix = '_(Woodwinds)_17_HP-Wind_Inst-UVR.wav'
        for relative_sax_loudness in [2, 5, 8]:
            for participant in range(1, 6):
                for song in range(1, 49):
                    id_ = f'FS{participant}_{song:02d}'
                    lines.append({
                        'example_id': id_,
                        'participant': participant,
                        'song': song,
                        'dataset': 'filosax',
                        'clean_solo': f'{args.uvr_filosax_path}/{id_}_L{relative_sax_loudness}{suffix}',
                        'backing_pd': np.nan,
                        'backing_bd': np.nan,
                        'mix_path': np.nan,
                    })
            df = pd.DataFrame(lines)
            df.to_csv(f'{args.output_dir}/index_uvr_filosax_L{relative_sax_loudness}.csv', index=False)

    if args.dataset in ['omnibook', 'all']:
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
        df.to_csv(f'{args.output_dir}/index_omnibook.csv', index=False)