import numpy as np
import pyloudnorm as pyln
from typing import Tuple
#from audio_separator.separator import Separator
import pandas as pd
import soundfile as sf
import argparse
import os

def mix_tracks_with_loudness(
    background: np.ndarray,
    sax: np.ndarray,
    sr: int,
    target_lufs: float = -23.0,
    sax_louder_by: float = 6.0
) -> np.ndarray:
    """
    Mix background and sax tracks with sax louder by a specified LUFS difference.

    Args:
        background: Background track (mono or stereo).
        sax: Saxophone track (same shape as background).
        sr: Sampling rate.
        target_lufs: Final integrated LUFS for the full mix.
        sax_louder_by: How much louder (in LUFS) the sax should be relative to the background.

    Returns:
        np.ndarray: Mixed audio signal normalized to target LUFS.
    """
    meter = pyln.Meter(sr)

    bg_loudness = meter.integrated_loudness(background)
    sax_loudness = meter.integrated_loudness(sax)

    # Adjust background to reference loudness
    background_adjusted = pyln.normalize.loudness(background, bg_loudness, target_lufs)
    sax_adjusted = pyln.normalize.loudness(sax, sax_loudness, target_lufs + sax_louder_by)

    # Mix the two
    mix = background_adjusted + sax_adjusted

    # Normalize final mix
    mix_loudness = meter.integrated_loudness(mix)
    final_mix = pyln.normalize.loudness(mix, mix_loudness, target_lufs)

    return final_mix


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default='tmp')
    args = parser.parse_args()

    #separator = Separator(model_file_dir=args.model_dir, output_dir=args.output_dir, sample_rate = 44100, output_single_stem='Woodwinds')
    #separator.load_model('17_HP-Wind_Inst-UVR.pth')

    os.makedirs(args.output_dir, exist_ok=True)

    index_file = pd.read_csv(args.dataset_index)
    for _, elem in index_file.iterrows():
        left, srl = sf.read(elem['backing_pd'])
        right, srr = sf.read(elem['backing_bd'])
        solo, srs = sf.read(elem['clean_solo'])
        assert srl == srr == srs == 44100
        background = (left + right) / 2
        for relative_sax_loudness in [2, 5, 8]:
            mix = mix_tracks_with_loudness(background, solo, srs, target_lufs=-23.0, sax_louder_by=relative_sax_loudness)
            filename = f'{args.output_dir}/{elem["example_id"]}_L{relative_sax_loudness}.wav'
            sf.write(filename, mix, srs)
            #output_files = separator.separate(filename)