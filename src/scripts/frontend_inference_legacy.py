import numpy as np
from src.dataset.frontend_dataset import FilosaxFrontendTrackDataset, FrontendSegmentDataset, FrontendTrackDataset
from torch.utils.data import DataLoader
from src.model.crnn_frontend import MusicTranscriptionLightning
import argparse
import os

"""
    calculate input features (CRNN) for rhythm perceiver: datasets filosax and wjd
"""

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--melspec_dir', type=str, required=True)
    parser.add_argument('--frame_labels_dir', type=str, required=True)
    parser.add_argument('--frames', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    filosax_l8 = FilosaxFrontendTrackDataset(data_dir_x=args.melspec_dir + '/uvr_filosax_L8', data_dir_y=args.frame_labels_dir + '/filosax', split = 'all', min_pitch_shift = -3, max_pitch_shift = 8)
    filosax_l5 = FilosaxFrontendTrackDataset(data_dir_x=args.melspec_dir + '/uvr_filosax_L5', data_dir_y=args.frame_labels_dir + '/filosax', split = 'all', min_pitch_shift = -3, max_pitch_shift = 8)
    filosax_l2 = FilosaxFrontendTrackDataset(data_dir_x=args.melspec_dir + '/uvr_filosax_L2', data_dir_y=args.frame_labels_dir + '/filosax', split = 'all', min_pitch_shift = -3, max_pitch_shift = 8)
    wjd = FrontendTrackDataset(data_dir_x=args.melspec_dir + '/wjd', data_dir_y=args.frame_labels_dir + '/wjd', min_pitch_shift = -3, max_pitch_shift = 3)

    model = MusicTranscriptionLightning(
        mel_bins=229,
        classes_num=88,
        learning_rate=1e-3,
        weight_decay=1e-4,
        var = 1
    )

    # Load the weights from your PyTorch checkpoint
    model_path = args.checkpoint
    model.load_weights_from_pytorch_checkpoint(model_path)
    model.eval()
    model.to('cuda');

    if args.model_type == 'filosax_l8':
        dataset = filosax_l8
        output_folder = args.output_dir + '/uvr_filosax_L8'
    elif args.model_type == 'filosax_l5':
        dataset = filosax_l5
        output_folder = args.output_dir + '/uvr_filosax_L5'
    elif args.model_type == 'filosax_l2':
        dataset = filosax_l2
        output_folder = args.output_dir + '/uvr_filosax_L2'
    elif args.model_type == 'wjd':
        dataset = wjd
        output_folder = args.output_dir + '/wjd'
    else:
        raise ValueError(f'Invalid model type: {args.model_type}')

    for track in dataset:
        segments = FrontendSegmentDataset([track], num_consecutive_frames = args.frames, use_cache = False)
        loader = DataLoader(segments, batch_size=args.batch_size, shuffle=False, num_workers=0)
        onsets = []
        offsets = []
        frames = []
        for batch in loader:
            output = model(batch['x']['mel_spec'].to('cuda'))
            onsets.append(output['reg_onset_output'].to('cpu').detach())
            offsets.append(output['reg_offset_output'].to('cpu').detach())
            frames.append(output['frame_output'].to('cpu').detach())
        frames = np.concatenate(frames).reshape(-1, 88)[:track['mel_spec'].shape[0]]
        onsets = np.concatenate(onsets).reshape(-1, 88)[:track['mel_spec'].shape[0]]
        offsets = np.concatenate(offsets).reshape(-1, 88)[:track['mel_spec'].shape[0]]
        np.save(os.path.join(output_folder, track['id'] + '.frames.npy'), frames)
        np.save(os.path.join(output_folder, track['id'] + '.onsets.npy'), onsets)
        np.save(os.path.join(output_folder, track['id'] + '.offsets.npy'), offsets)