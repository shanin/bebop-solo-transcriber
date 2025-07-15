from src.dataset.dataset import FilosaxDataset, SegmentDataset
from torch.utils.data import DataLoader
from src.model.model_a import RhythmScaffoldLightningModule
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import ConcatDataset

def main(args):

    filosax_easy = args.filosax_dir.replace('filosax', 'uvr_filosax_L8')
    filosax_med = args.filosax_dir.replace('filosax', 'uvr_filosax_L5')
    filosax_hard = args.filosax_dir.replace('filosax', 'uvr_filosax_L2')

    filosax_train_easy = FilosaxDataset(data_dir=filosax_easy, split = 'train', source = 'original')
    filosax_train_segments_easy = SegmentDataset(filosax_train_easy, num_consecutive_bars = 8, random_transposition = args.transpose_augmentation)

    filosax_train_med = FilosaxDataset(data_dir=filosax_med, split = 'train', source = 'original')
    filosax_train_segments_med = SegmentDataset(filosax_train_med, num_consecutive_bars = 8, random_transposition = args.transpose_augmentation)

    filosax_val_med = FilosaxDataset(data_dir=filosax_med, split = 'val', source = 'original')
    filosax_val_segments_med = SegmentDataset(filosax_val_med, num_consecutive_bars = 8, random_transposition = False)
    filosax_val_loader = DataLoader(filosax_val_segments_med, batch_size=512, shuffle=False, num_workers=0)

    filosax_test_med = FilosaxDataset(data_dir=filosax_med, split = 'test', source = 'original')
    filosax_test_segments_med = SegmentDataset(filosax_test_med, num_consecutive_bars = 8, random_transposition = False)
    filosax_test_loader = DataLoader(filosax_test_segments_med, batch_size=512, shuffle=False, num_workers=0)

    filosax_train = ConcatDataset([filosax_train_segments_easy, filosax_train_segments_med])
    len_easy = len(filosax_train_segments_easy)
    len_med = len(filosax_train_segments_med)

    # Index which sample comes from which dataset
    source_ids = [0] * len_easy + [1] * len_med  # 0 = A, 1 = B
    # Desired sampling ratio
    weight_a = 0.5
    weight_b = 0.5

    weights = [weight_a] * len_easy + [weight_b] * len_med

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),  # or more if oversampling
        replacement=True
    )

    filosax_train_loader = DataLoader(
        filosax_train,
        batch_size=512,
        num_workers=0,
        sampler=sampler
    )

    model = RhythmScaffoldLightningModule(
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        teacher_forcing=True,
        project_name="solo-transcriber",
        experiment_name = args.experiment_name
    )

    if args.starting_checkpoint is not None:
        model.load_from_checkpoint(args.starting_checkpoint)

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
    parser.add_argument("--starting_checkpoint", type=str, default=None)
    args = parser.parse_args()
    main(args)