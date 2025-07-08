
import torch
from src.dataset.dataset import dataloader_generator
from src.dataset.dataset import FilosaxDataset
import argparse
from src.model.model_a import RhythmScaffoldLightningModule
import pytorch_lightning as pl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='solo-transcriber')
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    model = RhythmScaffoldLightningModule(
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        teacher_forcing=True,
        project_name=args.project_name
    )

    trainer = pl.Trainer(
        max_epochs=100,
        logger=model.wandb_logger,
        callbacks=model.get_trainer_callbacks(),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )

    model.project_name = args.project_name
    filosax_train = FilosaxDataset(data_dir=args.data_dir, split = 'train', source = 'original')
    filosax_val = FilosaxDataset(data_dir=args.data_dir, split = 'val', source = 'original')
    filosax_train_loader = dataloader_generator(
        filosax_train, 
        num_consecutive_bars = 8,
        random_transposition = True,
        mode = 'tenor',
    )
    filosax_val_loader = dataloader_generator(
        filosax_val, 
        num_consecutive_bars = 8,
        random_transposition = True,
        mode = 'tenor',
    )
    trainer.fit(model, filosax_train_loader, filosax_val_loader)