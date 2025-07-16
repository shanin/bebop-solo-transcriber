import pandas as pd
import numpy as np
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['filosax', 'uvr_filosax', 'omnibook', 'all', 'inference'], default='all')
    parser.add_argument('--filosax_path', type=str)
    parser.add_argument('--uvr_filosax_path', type=str)
    parser.add_argument('--omnibook_path', type=str)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    if args.dataset == 'inference':
        assert args.data_path is not None, 'data_path is required'
        lines = []
        files = os.listdir(args.data_path)
        for file in files:
            if file.endswith('.wav'):
                id_ = file.split('.')[0]
                lines.append({
                    'example_id': id_,
                    'dataset': 'inference',
                    'clean_solo': f'{args.data_path}/{file}',
                    'backing_pd': np.nan,
                    'backing_bd': np.nan,
                    'mix_path': np.nan,
                })
        df = pd.DataFrame(lines)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        df.to_csv(f'{args.output_dir}/index_inference.csv', index=False)

    if args.dataset in ['filosax', 'all']:
        assert args.filosax_path is not None, 'filosax_path is required'
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
        assert args.uvr_filosax_path is not None, 'uvr_filosax_path is required'
        suffix = '_(Woodwinds)_17_HP-Wind_Inst-UVR.wav'
        for relative_sax_loudness in [2, 5, 8]:
            lines = []
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
        assert args.omnibook_path is not None, 'omnibook_path is required'
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