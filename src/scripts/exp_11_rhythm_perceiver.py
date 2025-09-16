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
    
    train_dataset = FilosaxBackendDataset(bars_dir=args.bars_dir, crnn_dir=args.crnn_dir, posenc_dir=args.posenc_dir, split='train', source='original')
    train_dataset_dt = FilosaxBackendDataset(bars_dir=args.bars_dir, crnn_dir=args.crnn_dir, posenc_dir=args.posenc_dir, split='train', source='double_time')
    train_dataset_dts = FilosaxBackendDataset(bars_dir=args.bars_dir, crnn_dir=args.crnn_dir, posenc_dir=args.posenc_dir, split='train', source='double_time_shifted')
    train_segments = BackendSegmentDataset(train_dataset, num_consecutive_bars = args.num_consecutive_bars)
    train_segments_dt = BackendSegmentDataset(train_dataset_dt, num_consecutive_bars = args.num_consecutive_bars)
    train_segments_dts = BackendSegmentDataset(train_dataset_dts, num_consecutive_bars = args.num_consecutive_bars)
    val_dataset = FilosaxBackendDataset(bars_dir=args.bars_dir, crnn_dir=args.crnn_dir, posenc_dir=args.posenc_dir, split='val')
    val_segments = BackendSegmentDataset(val_dataset, num_consecutive_bars = args.num_consecutive_bars)
    test_dataset = FilosaxBackendDataset(bars_dir=args.bars_dir, crnn_dir=args.crnn_dir, posenc_dir=args.posenc_dir, split='test')
    test_segments = BackendSegmentDataset(test_dataset, num_consecutive_bars = args.num_consecutive_bars)

    if not args.double_time_augmentation:
        train_loader = DataLoader(train_segments, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=backend_segment_collate_fn)
    else:
        weights = [1.0] * len(train_segments) + [.5] * len(train_segments_dt) + [.5] * len(train_segments_dts)
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(weights), 
            replacement=True
        )
        train_loader = DataLoader(
            ConcatDataset([train_segments, train_segments_dt, train_segments_dts]), 
            batch_size=args.batch_size,  
            num_workers=0, 
            collate_fn=backend_segment_collate_fn, 
            sampler=sampler
        )
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
        label_smoothing=args.label_smoothing,
        disable_rhythm_classifier=args.disable_rhythm_classifier,
        disable_structural_injection=args.disable_structural_injection,
    )

    # Create trainer with proper WandbLogger and memory optimizations
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision="32",
        check_val_every_n_epoch=1,
        log_every_n_steps=100,
        enable_checkpointing=True,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                filename=f'{args.experiment_name}-{{epoch:02d}}-{{val_loss:.4f}}',
                auto_insert_metric_name=False,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                mode="min",
                verbose=True,
            ),
        ],
        logger=model.wandb_logger,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_consecutive_bars", type=int, default=8, help="Number of consecutive bars per segment")

    # Training arguments
    parser.add_argument("--experiment_name", type=str, default="exp_11_rhythm_perceiver", help="W&B experiment name")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--rhythm_loss_weight", type=float, default=1.0, help="Rhythm loss weight")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num_backend_heads", type=int, default=8, help="Number of backend attention heads")
    parser.add_argument("--num_backend_layers", type=int, default=2, help="Number of backend layers")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing factor")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--double_time_augmentation", action='store_true', default=False)
    parser.add_argument("--disable_rhythm_classifier", action='store_true', default=False)
    parser.add_argument("--disable_structural_injection", action='store_true', default=False)
    
    args = parser.parse_args()
    
    print(f"Starting experiment: {args.experiment_name}")
    print(f"Batch size: {args.batch_size}, Consecutive bars: {args.num_consecutive_bars}")
    print(f"Model - Embedding dim: {args.embedding_dim}, Heads: {args.num_heads}, Layers: {args.num_layers}")
    print(f"Learning rate: {args.learning_rate}, Rhythm loss weight: {args.rhythm_loss_weight}")
    
    main(args)