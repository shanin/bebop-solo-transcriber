
""" adapted from https://github.com/bytedance/piano_transcription """

import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torchlibrosa.stft import Spectrogram, LogmelFilterBank


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        
        return x


class AcousticModelCRnn8Dropout(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(AcousticModelCRnn8Dropout, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128, momentum=momentum)

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)

        self.fc = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, time_steps, classes_num)
        """

        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=False) #Modify inplace to False
        
        (x, _) = self.gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        output = torch.sigmoid(self.fc(x))
        return output


class Regress_onset_offset_frame_velocity_CRNN(nn.Module):
    def __init__(self, mel_bins, classes_num):
        super(Regress_onset_offset_frame_velocity_CRNN, self).__init__()

        midfeat = 1792
        momentum = 0.01

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)

        self.frame_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_onset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.reg_offset_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)
        self.velocity_model = AcousticModelCRnn8Dropout(classes_num, midfeat, momentum)

        self.reg_onset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(input_size=88 * 3, hidden_size=256, num_layers=1, 
            bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)
 
    def forward(self, x):
        """
        Args:
          x: (batch_size, time_steps, mel_bins = 229)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """

        x = x.unsqueeze(1)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        frame_output = self.frame_model(x)  # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_offset_model(x)    # (batch_size, time_steps, classes_num)
        velocity_output = self.velocity_model(x)    # (batch_size, time_steps, classes_num)
 
        # Use velocities to condition onset regression
        #x = torch.cat((reg_onset_output, velocity_output.detach()), dim=2)
        #(x, _) = self.reg_onset_gru(x)
        #x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        #reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        """(batch_size, time_steps, classes_num)"""

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)
        frame_output = torch.sigmoid(self.frame_fc(x))  # (batch_size, time_steps, classes_num)
        """(batch_size, time_steps, classes_num)"""

        output_dict = {
            'reg_onset_output': reg_onset_output, 
            'reg_offset_output': reg_offset_output, 
            'frame_output': frame_output, 
            'velocity_output': velocity_output}

        return output_dict


class MusicTranscriptionLightning(pl.LightningModule):
    """
    PyTorch Lightning module for music transcription using CRNN with onset/offset regression.
    """
    
    def __init__(
        self,
        mel_bins=229,
        classes_num=88,
        learning_rate=1e-3,
        weight_decay=1e-4,
        onset_loss_weight=1.0,
        offset_loss_weight=1.0,
        frame_loss_weight=1.0,
        velocity_loss_weight=1.0,
        scheduler_patience=5,
        scheduler_factor=0.5
    ):
        super().__init__()
        
        # Optimize for A100 Tensor Cores
        torch.set_float32_matmul_precision('medium')
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model
        self.model = Regress_onset_offset_frame_velocity_CRNN(
            mel_bins=mel_bins, 
            classes_num=classes_num
        )
        
        # Loss weights
        self.onset_loss_weight = onset_loss_weight
        self.offset_loss_weight = offset_loss_weight
        self.frame_loss_weight = frame_loss_weight
        self.velocity_loss_weight = velocity_loss_weight
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def _calculate_losses(self, outputs, targets):
        """Calculate all loss components."""
        
        # Extract predictions
        onset_pred = outputs['reg_onset_output']      # (batch, time, 88)
        offset_pred = outputs['reg_offset_output']    # (batch, time, 88)
        frame_pred = outputs['frame_output']          # (batch, time, 88)
        velocity_pred = outputs['velocity_output']    # (batch, time, 88)
        
        # Extract targets
        onset_target = targets['onset']     # (batch, time, 88)
        offset_target = targets['offset']   # (batch, time, 88)
        frame_target = targets['frames']    # (batch, time, 88)
        
        # Binary cross-entropy for onset/offset regression targets
        onset_loss = F.binary_cross_entropy(onset_pred, onset_target)
        offset_loss = F.binary_cross_entropy(offset_pred, offset_target)
        
        # Binary cross-entropy for frame-wise classification
        frame_loss = F.binary_cross_entropy(frame_pred, frame_target)
        
        # Binary cross-entropy for velocity prediction
        # Create velocity targets based on frame activity
        velocity_target = frame_target  # Simple approach: velocity = frame activity
        velocity_loss = F.binary_cross_entropy(velocity_pred, velocity_target)
        
        # Total weighted loss
        total_loss = (
            self.onset_loss_weight * onset_loss +
            self.offset_loss_weight * offset_loss +
            self.frame_loss_weight * frame_loss +
            self.velocity_loss_weight * velocity_loss
        )
        
        return {
            'total_loss': total_loss,
            'onset_loss': onset_loss,
            'offset_loss': offset_loss,
            'frame_loss': frame_loss,
            'velocity_loss': velocity_loss
        }
    
    def _calculate_metrics(self, outputs, targets):
        """Calculate evaluation metrics."""
        
        # Extract predictions and targets
        frame_pred = outputs['frame_output']
        frame_target = targets['frames']
        
        # Convert to binary predictions (threshold at 0.5)
        frame_pred_binary = (frame_pred > 0.5).float()
        
        # Calculate frame-wise metrics
        true_positives = (frame_pred_binary * frame_target).sum()
        predicted_positives = frame_pred_binary.sum()
        actual_positives = frame_target.sum()
        
        # Precision, Recall, F1
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Accuracy
        correct = (frame_pred_binary == frame_target).float().sum()
        total = frame_target.numel()
        accuracy = correct / total
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        
        # Unpack batch
        mel_spec = batch['x']['mel_spec']    # (batch, 1, time, mel_bins)
        targets = {
            'onset': batch['y']['onset'],    # (batch, time, 88)
            'offset': batch['y']['offset'],  # (batch, time, 88)
            'frames': batch['y']['frames']   # (batch, time, 88)
        }
        
        # Forward pass with gradient checkpointing to save memory
        #with torch.cuda.amp.autocast():  # Use mixed precision
        #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = self(mel_spec)
        
        # Calculate losses
        losses = self._calculate_losses(outputs, targets)
        
        # Log losses
        self.log('train_loss', losses['total_loss'], prog_bar=True)
        self.log('train_onset_loss', losses['onset_loss'])
        self.log('train_offset_loss', losses['offset_loss'])
        self.log('train_frame_loss', losses['frame_loss'])
        self.log('train_velocity_loss', losses['velocity_loss'])
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        
        # Unpack batch
        mel_spec = batch['x']['mel_spec']
        targets = {
            'onset': batch['y']['onset'],
            'offset': batch['y']['offset'],
            'frames': batch['y']['frames']
        }
        
        # Forward pass
        outputs = self(mel_spec)
        
        # Calculate losses
        losses = self._calculate_losses(outputs, targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(outputs, targets)
        
        # Log validation metrics
        self.log('val_loss', losses['total_loss'], prog_bar=True)
        self.log('val_onset_loss', losses['onset_loss'])
        self.log('val_offset_loss', losses['offset_loss'])
        self.log('val_frame_loss', losses['frame_loss'])
        self.log('val_velocity_loss', losses['velocity_loss'])
        
        self.log('val_precision', metrics['precision'])
        self.log('val_recall', metrics['recall'])
        self.log('val_f1', metrics['f1'], prog_bar=True)
        self.log('val_accuracy', metrics['accuracy'])
        
        return losses['total_loss']
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        
        # Unpack batch
        mel_spec = batch['x']['mel_spec']
        targets = {
            'onset': batch['y']['onset'],
            'offset': batch['y']['offset'],
            'frames': batch['y']['frames']
        }
        
        # Forward pass
        outputs = self(mel_spec)
        
        # Calculate losses
        losses = self._calculate_losses(outputs, targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(outputs, targets)
        
        # Log test metrics
        self.log('test_loss', losses['total_loss'])
        self.log('test_onset_loss', losses['onset_loss'])
        self.log('test_offset_loss', losses['offset_loss'])
        self.log('test_frame_loss', losses['frame_loss'])
        self.log('test_velocity_loss', losses['velocity_loss'])
        
        self.log('test_precision', metrics['precision'])
        self.log('test_recall', metrics['recall'])
        self.log('test_f1', metrics['f1'])
        self.log('test_accuracy', metrics['accuracy'])
        
        return {
            'test_loss': losses['total_loss'],
            'test_f1': metrics['f1'],
            'outputs': outputs,
            'targets': targets
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        
        # Optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.scheduler_factor,
            patience=self.scheduler_patience
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        
        mel_spec = batch['x']['mel_spec']
        outputs = self(mel_spec)
        
        return {
            'onset_pred': outputs['reg_onset_output'],
            'offset_pred': outputs['reg_offset_output'],
            'frame_pred': outputs['frame_output'],
            'velocity_pred': outputs['velocity_output']
        }
    
    @staticmethod
    def create_wandb_logger(project_name="bebop-solo-transcriber", experiment_name="crnn_training", save_dir="wandb_logs/"):
        """
        Create a properly configured WandbLogger for PyTorch Lightning.
        
        Args:
            project_name (str): W&B project name
            experiment_name (str): W&B run name
            save_dir (str): Local directory to save logs
            
        Returns:
            WandbLogger: Configured logger for PyTorch Lightning
        """
        return WandbLogger(
            project=project_name,
            name=experiment_name,
            save_dir=save_dir,
            offline=False,
            log_model=True,  # Log model checkpoints to W&B
            tags=["crnn", "music-transcription", "onset-offset", "regression"],
            notes="Training CRNN with regression-based onset/offset detection for music transcription"
        )
    
    @staticmethod
    def create_trainer_with_wandb(
        project_name="bebop-solo-transcriber", 
        experiment_name="crnn_training",
        max_epochs=100,
        patience=10
    ):
        """
        Create a complete trainer setup with WandbLogger and callbacks.
        
        Args:
            project_name (str): W&B project name
            experiment_name (str): W&B run name  
            max_epochs (int): Maximum training epochs
            patience (int): Early stopping patience
            
        Returns:
            tuple: (trainer, wandb_logger) configured for training
        """
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        
        # Create WandbLogger
        wandb_logger = MusicTranscriptionLightning.create_wandb_logger(
            project_name=project_name,
            experiment_name=experiment_name
        )
        
        # Callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            verbose=True
        )
        
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            filename="best-model-{epoch:02d}-{val_loss:.2f}",
            save_last=True
        )
        
        # Trainer with memory optimizations
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            logger=wandb_logger,  # Use proper WandbLogger
            callbacks=[early_stop_callback, checkpoint_callback],
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            enable_progress_bar=True
        )
        
        return trainer, wandb_logger