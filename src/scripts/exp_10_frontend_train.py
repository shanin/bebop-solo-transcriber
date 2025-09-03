from src.dataset.frontend_dataset import FilosaxFrontendTrackDataset, FrontendSegmentDataset, FrontendTrackDataset
from torch.utils.data import DataLoader
from src.model.crnn_frontend import MusicTranscriptionLightning
import pytorch_lightning as pl
import torch
import argparse
import os
import wandb
from torch.utils.data import ConcatDataset, WeightedRandomSampler


def main(args):
    # Set tensor core optimization for A100 GPU
    torch.set_float32_matmul_precision('medium')

    filosax_dir_x_L2 = os.path.join(args.data_dir_x, 'uvr_filosax_L2')
    filosax_dir_x_L5 = os.path.join(args.data_dir_x, 'uvr_filosax_L5')
    filosax_dir_x_L8 = os.path.join(args.data_dir_x, 'uvr_filosax_L8')
    filosax_dir_y = os.path.join(args.data_dir_y, 'filosax')

    wjd_dir_x = os.path.join(args.data_dir_x, 'wjd')
    wjd_dir_y = os.path.join(args.data_dir_y, 'wjd')

    filosax_train_L2 = FilosaxFrontendTrackDataset(data_dir_x=filosax_dir_x_L2, data_dir_y=filosax_dir_y, split = 'train', min_pitch_shift = -3, max_pitch_shift = 8)
    filosax_train_segments_L2 = FrontendSegmentDataset(filosax_train_L2, num_consecutive_frames = args.frames, use_cache = False)

    filosax_train_L5 = FilosaxFrontendTrackDataset(data_dir_x=filosax_dir_x_L5, data_dir_y=filosax_dir_y, split = 'train', min_pitch_shift = -3, max_pitch_shift = 8)
    filosax_train_segments_L5 = FrontendSegmentDataset(filosax_train_L5, num_consecutive_frames = args.frames, use_cache = False)

    #filosax_train_L8 = FilosaxFrontendTrackDataset(data_dir_x=filosax_dir_x_L8, data_dir_y=filosax_dir_y, split = 'train', min_pitch_shift = 0, max_pitch_shift = 0)
    #filosax_train_segments_L8 = FrontendSegmentDataset(filosax_train_L8, num_consecutive_frames = args.frames, use_cache = False)

    filosax_val = FilosaxFrontendTrackDataset(data_dir_x=filosax_dir_x_L8, data_dir_y=filosax_dir_y, split = 'val')
    filosax_val_segments = FrontendSegmentDataset(filosax_val, num_consecutive_frames = args.frames, use_cache = False)
    filosax_val_loader = DataLoader(filosax_val_segments, batch_size=args.batch_size, shuffle=False, num_workers=0)

    filosax_test = FilosaxFrontendTrackDataset(data_dir_x=filosax_dir_x_L8, data_dir_y=filosax_dir_y, split = 'test')
    filosax_test_segments = FrontendSegmentDataset(filosax_test, num_consecutive_frames = args.frames, use_cache = False)
    filosax_test_loader = DataLoader(filosax_test_segments, batch_size=args.batch_size, shuffle=False, num_workers=0)

    wjd_train = FrontendTrackDataset(data_dir_x=wjd_dir_x, data_dir_y=wjd_dir_y, min_pitch_shift = -3, max_pitch_shift = 3)
    wjd_train_segments = FrontendSegmentDataset(wjd_train, num_consecutive_frames = args.frames, use_cache = False)

    dataset_train = ConcatDataset([filosax_train_segments_L5, filosax_train_segments_L2, wjd_train_segments])
    len_filosax_L5 = len(filosax_train_segments_L5)
    len_filosax_L2 = len(filosax_train_segments_L2)
    len_wjd = len(wjd_train_segments)

    weight_filosax = 0.4
    weight_wjd = 0.6
    weights = [weight_filosax / 2] * len_filosax_L5 + [weight_filosax / 2] * len_filosax_L2 + [weight_wjd] * len_wjd

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights), 
        replacement=True
    )

    filosax_train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=0,
        sampler=sampler
    )

    # Create model with configurable hyperparameters
    model = MusicTranscriptionLightning(
        mel_bins=229, 
        classes_num=88,
        learning_rate=args.learning_rate,
        onset_loss_weight=args.onset_weight,
        offset_loss_weight=args.offset_weight,
        frame_loss_weight=args.frame_weight,
        velocity_loss_weight=args.velocity_weight
    )

    # Create trainer with proper WandbLogger and memory optimizations
    trainer, wandb_logger = MusicTranscriptionLightning.create_trainer_with_wandb(
        project_name="bebop-solo-transcriber",
        experiment_name=args.experiment_name,
        max_epochs=args.max_epochs,
        patience=args.patience
    )

    # Train the model
    trainer.fit(model, filosax_train_loader, filosax_val_loader)

    # Test the model
    trainer.test(model, filosax_test_loader)

    # Print best model path
    checkpoint_callback = None
    for callback in trainer.callbacks:
        if hasattr(callback, 'best_model_path'):
            checkpoint_callback = callback
            break
    
    if checkpoint_callback:
        print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    
    # Finish W&B logging
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CRNN frontend for music transcription")
    
    # Data arguments
    parser.add_argument("--data_dir_x", type=str, required=True, help="Directory for mel spectrogram data")
    parser.add_argument("--data_dir_y", type=str, required=True, help="Directory for frame labels data")
    parser.add_argument("--frames", type=int, default=500, help="Number of consecutive frames per segment")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    # Training arguments
    parser.add_argument("--experiment_name", type=str, default="exp_10_frontend_train", help="W&B experiment name")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Loss weight arguments
    parser.add_argument("--onset_weight", type=float, default=1.0, help="Onset loss weight")
    parser.add_argument("--offset_weight", type=float, default=1.0, help="Offset loss weight")
    parser.add_argument("--frame_weight", type=float, default=1.0, help="Frame loss weight")
    parser.add_argument("--velocity_weight", type=float, default=0.5, help="Velocity loss weight")
    
    args = parser.parse_args()
    
    print(f"Starting experiment: {args.experiment_name}")
    print(f"Batch size: {args.batch_size}, Frames: {args.frames}")
    print(f"Loss weights - Onset: {args.onset_weight}, Offset: {args.offset_weight}, Frame: {args.frame_weight}, Velocity: {args.velocity_weight}")
    
    main(args)
