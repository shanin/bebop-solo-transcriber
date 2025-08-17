from src.dataset.frontend_dataset import FilosaxFrontendTrackDataset, FrontendSegmentDataset, FrontendTrackDataset
from torch.utils.data import DataLoader
from src.model.crnn_frontend import MusicTranscriptionLightning as frontend
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import os
import wandb
from torch.utils.data import ConcatDataset, WeightedRandomSampler


def main(args):
    filosax_dir_x = os.path.join(args.data_dir_x, 'filosax')
    filosax_dir_y = os.path.join(args.data_dir_y, 'filosax')
    wjd_dir_x = os.path.join(args.data_dir_x, 'wjd')
    wjd_dir_y = os.path.join(args.data_dir_y, 'wjd')

    filosax_train = FilosaxFrontendTrackDataset(data_dir_x=filosax_dir_x, data_dir_y=filosax_dir_y, split = 'train')
    filosax_train_segments = FrontendSegmentDataset(filosax_train, num_consecutive_frames = args.frames, use_cache = False)

    filosax_val = FilosaxFrontendTrackDataset(data_dir_x=filosax_dir_x, data_dir_y=filosax_dir_y, split = 'val')
    filosax_val_segments = FrontendSegmentDataset(filosax_val, num_consecutive_frames = args.frames, use_cache = False)
    filosax_val_loader = DataLoader(filosax_val_segments, batch_size=args.batch_size, shuffle=False, num_workers=0)

    filosax_test = FilosaxFrontendTrackDataset(data_dir_x=filosax_dir_x, data_dir_y=filosax_dir_y, split = 'test')
    filosax_test_segments = FrontendSegmentDataset(filosax_test, num_consecutive_frames = args.frames, use_cache = False)
    filosax_test_loader = DataLoader(filosax_test_segments, batch_size=args.batch_size, shuffle=False, num_workers=0)

    wjd_train = FrontendTrackDataset(data_dir_x=wjd_dir_x, data_dir_y=wjd_dir_y)
    wjd_train_segments = FrontendSegmentDataset(wjd_train, num_consecutive_frames = args.frames, use_cache = False)

    dataset_train = ConcatDataset([filosax_train_segments, wjd_train_segments])
    len_filosax = len(filosax_train_segments)
    len_wjd = len(wjd_train_segments)

    weight_filosax = 0.5
    weight_wjd = 0.5
    weights = [weight_filosax] * len_filosax + [weight_wjd] * len_wjd

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights), 
        replacement=True
    )

    filosax_train_loader = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=0,
        sampler=sampler
    )

    model = frontend(mel_bins=229, classes_num=88)
    model.train()

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

    wandb_logger = wandb.init(project="frontend-train", name=args.experiment_name)

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    trainer.fit(model, filosax_train_loader, filosax_val_loader)

    trainer.fit(model, filosax_train_loader, filosax_val_loader)

    trainer.test(model, filosax_test_loader)

    wandb_logger.finish()

    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--data_dir_x", type=str, required=True)
    parser.add_argument("--data_dir_y", type=str, required=True)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
