from src.dataset.dataset import FilosaxDataset, SegmentDataset
from torch.utils.data import DataLoader
from src.model.model_a import RhythmScaffoldLightningModule
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse

def main(args):
    filosax_train = FilosaxDataset(data_dir=args.filosax_dir, split = 'train', source = 'original')
    filosax_train_segments = SegmentDataset(filosax_train, num_consecutive_bars = 8, random_transposition = args.transpose_augmentation)
    filosax_train_loader = DataLoader(filosax_train_segments, batch_size=512, shuffle=True, num_workers=0)

    filosax_val = FilosaxDataset(data_dir=args.filosax_dir, split = 'val', source = 'original')
    filosax_val_segments = SegmentDataset(filosax_val, num_consecutive_bars = 8, random_transposition = False)
    filosax_val_loader = DataLoader(filosax_val_segments, batch_size=512, shuffle=False, num_workers=0)

    filosax_test = FilosaxDataset(data_dir=args.filosax_dir, split = 'test', source = 'original')
    filosax_test_segments = SegmentDataset(filosax_test, num_consecutive_bars = 8, random_transposition = False)
    filosax_test_loader = DataLoader(filosax_test_segments, batch_size=512, shuffle=False, num_workers=0)

    model = RhythmScaffoldLightningModule(
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        teacher_forcing=True,
        project_name="solo-transcriber",
        experiment_name = args.experiment_name,
        rhythm_loss_weight = args.rhythm_loss_weight
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",     # or any other metric you log
        patience=5,             # how many epochs with no improvement to wait
        mode="min",             # "min" for loss, "max" for accuracy, etc.
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-model"
    )

    trainer = pl.Trainer(
        max_epochs=100,
        logger=model.wandb_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    trainer.fit(model, filosax_train_loader, filosax_val_loader)

    trainer.test(model, filosax_test_loader)

    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--filosax_dir", type=str, required=True)
    parser.add_argument("--transpose_augmentation", type=bool, default=True)
    parser.add_argument("--rhythm_loss_weight", type=float, default=1.0)
    args = parser.parse_args()
    main(args)