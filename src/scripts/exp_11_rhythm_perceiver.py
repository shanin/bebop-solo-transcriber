from src.dataset.backend_dataset import FilosaxBackendDataset, BackendSegmentDataset, backend_segment_collate_fn
from torch.utils.data import DataLoader
from src.model.rhythm_perceiver import RhythmPerceiverLightningModule
import pytorch_lightning as pl
import torch
import argparse
import os
import wandb
from torch.utils.data import ConcatDataset, WeightedRandomSampler


def main(args):
    # Set tensor core optimization for A100 GPU
    torch.set_float32_matmul_precision('medium')
    
    train_dataset = FilosaxBackendDataset(bars_dir=args.bars_dir, crnn_dir=args.crnn_dir, posenc_dir=args.posenc_dir, split='train')
    train_segments = BackendSegmentDataset(train_dataset, num_consecutive_bars = args.num_consecutive_bars)
    val_dataset = FilosaxBackendDataset(bars_dir=args.bars_dir, crnn_dir=args.crnn_dir, posenc_dir=args.posenc_dir, split='val')
    val_segments = BackendSegmentDataset(val_dataset, num_consecutive_bars = args.num_consecutive_bars)
    test_dataset = FilosaxBackendDataset(bars_dir=args.bars_dir, crnn_dir=args.crnn_dir, posenc_dir=args.posenc_dir, split='test')
    test_segments = BackendSegmentDataset(test_dataset, num_consecutive_bars = args.num_consecutive_bars)

    train_loader = DataLoader(train_segments, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=backend_segment_collate_fn)
    val_loader = DataLoader(val_segments, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=backend_segment_collate_fn)
    test_loader = DataLoader(test_segments, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=backend_segment_collate_fn)

    # Create model with configurable hyperparameters
    model = RhythmPerceiverLightningModule(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_backend_heads=args.num_backend_heads,
        num_backend_layers=args.num_backend_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        project_name="solo-transcriber",
        experiment_name=args.experiment_name,
        rhythm_loss_weight=args.rhythm_loss_weight,
    )

    # Create trainer with proper WandbLogger and memory optimizations
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision="32",
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        enable_checkpointing=True,
        checkpoint_monitor="val_loss",
        checkpoint_mode="min",
        checkpoint_save_top_k=3,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            )
        ],
        logger=wandb.init(
            project="solo-transcriber",
            name=args.experiment_name,
        )
    )

    # Train the model
    trainer.fit(model)
    trainer.test(model)

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
    parser.add_argument("--crnn_dir", type=str, required=True, help="Directory for CRNN data")
    parser.add_argument("--posenc_dir", type=str, required=True, help="Directory for position encoding data")
    parser.add_argument("--bars_dir", type=str, required=True, help="Directory for bars data")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_consecutive_bars", type=int, default=8, help="Number of consecutive bars per segment")

    # Training arguments
    parser.add_argument("--experiment_name", type=str, default="exp_11_rhythm_perceiver", help="W&B experiment name")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--rhythm_loss_weight", type=float, default=1.0, help="Rhythm loss weight")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_backend_heads", type=int, default=8, help="Number of backend attention heads")
    parser.add_argument("--num_backend_layers", type=int, default=2, help="Number of backend layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    args = parser.parse_args()
    
    print(f"Starting experiment: {args.experiment_name}")
    print(f"Batch size: {args.batch_size}, Frames: {args.frames}")
    print(f"Loss weights - Onset: {args.onset_weight}, Offset: {args.offset_weight}, Frame: {args.frame_weight}, Velocity: {args.velocity_weight}")
    
    main(args)
