import os
from src.dataset.dataset import  InferenceDataset, TrackDataset
from torch.utils.data import DataLoader
from src.model.model_a import RhythmScaffoldLightningModule
import torch
import argparse
from src.utils.renderer import main as render_midi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--handler', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    model = RhythmScaffoldLightningModule.load_from_checkpoint(f'solo-transcriber/{args.checkpoint}/checkpoints/best-model.ckpt')
    model.eval()
    model.to('cpu')

    test_track = TrackDataset(f'test/{args.track}/bars')
    test_data = InferenceDataset(test_track[0], num_consecutive_bars=8)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    result = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x']
            bin_logits, rhythm_logits = model(x)
            predictions = model._generate_structured_predictions(bin_logits.view(-1, 128), rhythm_logits.view(-1, 44))
            result.append(predictions)
    if test_data.last_segment == 0:
        result = torch.cat(result, dim=0)
    elif test_data.last_segment > 0:
        new_result = torch.cat(result[:-1], dim=0)
        result = torch.cat([new_result, result[-1][-test_data.last_segment * 48:]], dim=0)
    tokens = result

    output_dir = f'{args.output_dir}/{args.track}'
    os.makedirs(output_dir, exist_ok=True)

    render_midi(
        f'test/{args.track}/beats/{args.track}.beats.tsv', 
        tokens, 
        f'{output_dir}/{args.track}.{args.handler}.perf.mid', 
        f'{output_dir}/{args.track}.{args.handler}.score.mid', 
        f'test/{args.track}/raw/{args.track}.full.wav', 
        f'{output_dir}/{args.track}.{args.handler}.wav'
    )