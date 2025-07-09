import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from typing import Dict, Any, Optional, List, Tuple
import json

from src.tokenizer.rhythm_tokens import RHYTHM_TOKENS
INV_RHYTHM_TOKENS = {v['id']: k for k, v in RHYTHM_TOKENS.items()}

class PositionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        """
        Args:
            embedding_dim: Dimension of the output embeddings
        """
        super().__init__()
        
        # Beat embeddings (4 beats per bar)
        self.beat_embedding = nn.Embedding(4, embedding_dim)
        
        # Subdivision embeddings (12 subdivisions per beat + 1 for rhythm token)
        self.subdivision_embedding = nn.Embedding(13, embedding_dim)
        
        # Bar embeddings (for different bar positions in the sequence)
        self.bar_embedding = nn.Embedding(32, embedding_dim)  # Assuming max 100 bars per sequence
        
    def forward(self, x: dict) -> torch.Tensor:
        """
        Args:
            x: Dictionary containing:
                - bar_num: [batch, bars] tensor of bar numbers
                - features: [batch, bars, time, feature_dim] tensor of features
                
        Returns:
            torch.Tensor: Position embeddings of shape [batch, bars, time, embedding_dim]
        """
        batch_size, num_bars, seq_len, _ = x['features'].shape
        
        # Create beat indices (0-3 for each beat)
        beat_indices = torch.arange(seq_len, device=x['features'].device) // 12
        beat_indices = beat_indices.unsqueeze(0).unsqueeze(0)  # [1, 1, time]
        beat_indices = beat_indices.expand(batch_size, num_bars, -1)  # [batch, bars, time]
        
        # Create subdivision indices (0-11 for each subdivision)
        subdivision_indices = torch.arange(seq_len, device=x['features'].device) % 12
        subdivision_indices = subdivision_indices.unsqueeze(0).unsqueeze(0)  # [1, 1, time]
        subdivision_indices = subdivision_indices.expand(batch_size, num_bars, -1)  # [batch, bars, time]
        
        # Get relative bar indices (0 to num_bars-1)
        bar_indices = torch.arange(num_bars, device=x['features'].device)  # [bars]
        bar_indices = bar_indices.unsqueeze(0).unsqueeze(-1)  # [1, bars, 1] 
        bar_indices = bar_indices.expand(batch_size, -1, seq_len)  # [batch, bars, time]
        
        # Get embeddings
        beat_emb = self.beat_embedding(beat_indices)  # [batch, bars, time, embedding_dim]
        subdivision_emb = self.subdivision_embedding(subdivision_indices)  # [batch, bars, time, embedding_dim]
        bar_emb = self.bar_embedding(bar_indices)  # [batch, bars, time, embedding_dim]
        
        # Sum the embeddings
        position_emb = beat_emb + subdivision_emb + bar_emb  # [batch, bars, time, embedding_dim]

        # Rhythm position embeddings
        rhythm_beat_indices = torch.arange(seq_len // 12, device=x['features'].device) % 4
        rhythm_beat_indices = rhythm_beat_indices.unsqueeze(0).unsqueeze(0)
        rhythm_beat_indices = rhythm_beat_indices.expand(batch_size, num_bars, -1)
        rhythm_beat_emb = self.beat_embedding(rhythm_beat_indices)

        rhythm_bar_indices = torch.arange(seq_len // 12, device=x['features'].device) // 4
        rhythm_bar_indices = rhythm_bar_indices.unsqueeze(0).unsqueeze(0)
        rhythm_bar_indices = rhythm_bar_indices.expand(batch_size, num_bars, -1)
        rhythm_bar_emb = self.bar_embedding(rhythm_bar_indices)

        rhythm_subdivision_indices = torch.ones_like(rhythm_beat_indices) * 12
        rhythm_subdivision_emb = self.subdivision_embedding(rhythm_subdivision_indices)

        rhythm_position_emb = rhythm_beat_emb + rhythm_bar_emb + rhythm_subdivision_emb

        return position_emb, rhythm_position_emb

class JointPitchRhythmFeatureEncoder(nn.Module):
    def __init__(self, 
                 activation_dim: int = 384,
                 feature_dim: int = 3,  # flux, amplitude, confidence
                 embedding_dim: int = 128):
        """
        Args:
            activation_dim: Dimension of the activation input
            feature_dim: Dimension of the feature input (default: 3 for flux, amplitude, confidence)
            embedding_dim: Dimension of the output embeddings (default: 128)
        """
        super().__init__()
        
        # Activation encoder
        self.activation_encoder = nn.Sequential(
            nn.Linear(activation_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

        self.masked_rhythm_embedding = nn.Embedding(1, embedding_dim)
        
        # Position embedding
        self.position_bin_rhythm_embedding = PositionEmbedding(embedding_dim)
        
    def forward(self, x: dict) -> torch.Tensor:
        """
        Args:
            x: Dictionary containing:
                - activations: [batch, bars, time, activation_dim]
                - features: [batch, bars, time, feature_dim]
                - bar_num: [batch, bars] tensor of bar numbers
        
        Returns:
            torch.Tensor: Fused embeddings of shape [batch, bars, time, embedding_dim]
        """
        # Get inputs
        activations = x['x']['activations']  # [batch, bars, beats, bins, activation_dim]
        features = x['x']['features']  # [batch, bars, beats, feature_dim]

        # Get original shapes
        batch_size, num_bars, num_beats, num_bins, activation_dim = activations.shape
        _, _, _, features_dim, _ = features.shape
        activations = activations.view(batch_size, num_bars, num_beats * num_bins, activation_dim)
        features = features.transpose(-1, -2)
        features = features.view(batch_size, num_bars, num_beats * num_bins, features_dim)

        # Prepare masked rhythm embeddings
        # Create indices tensor filled with zeros (since we have only one embedding)
        rhythm_indices = torch.zeros((batch_size, num_bars, 4), dtype=torch.long, device=activations.device)
        
        # Get masked rhythm embeddings for each position
        masked_rhythm_emb = self.masked_rhythm_embedding(rhythm_indices)  # [batch, bars, beats, embedding_dim]
        
        # Reshape for linear layers
        activations = activations.view(-1, activations.shape[-1])  # [batch*bars*time, activation_dim]
        features = features.view(-1, features.shape[-1])  # [batch*bars*time, feature_dim]
        
        # Encode activations and features
        act_emb = self.activation_encoder(activations)  # [batch*bars*time, embedding_dim]
        feat_emb = self.feature_encoder(features)  # [batch*bars*time, embedding_dim]
        
        # Concatenate embeddings
        combined = torch.cat([act_emb, feat_emb], dim=-1)  # [batch*bars*time, embedding_dim*2]
        
        # Fuse embeddings
        fused = self.fusion(combined)  # [batch*bars*time, embedding_dim]
        
        # Reshape back to original dimensions
        fused = fused.view(batch_size, num_bars, num_beats * num_bins, -1)  # [batch, bars, time, embedding_dim]
        
        # Add position embeddings
        position_emb, rhythm_position_emb = self.position_bin_rhythm_embedding(x)  # [batch, bars, time, embedding_dim]
        fused = fused + position_emb  # [batch, bars, time, embedding_dim]
        rhythm = masked_rhythm_emb + rhythm_position_emb  # [batch, bars, beats, embedding_dim]

        return fused, rhythm

    
class JointPitchRhythmTransformerEncoder(nn.Module):
    def __init__(self, 
                 embedding_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 num_bin_classes: int = 128, 
                 num_rhythm_classes: int = 44,
                 rhythm_loss_weight: float = 1.0):
        """
        Args:
            embedding_dim: Dimension of the input embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            num_classes: Number of output classes (default: 129 for 128 pitches + rest)
        """
        super().__init__()
        
        # Feature encoder
        self.feature_encoder = JointPitchRhythmFeatureEncoder(embedding_dim=embedding_dim)
 
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.bin_output_projection = nn.Linear(embedding_dim, num_bin_classes)
        self.rhythm_output_projection = nn.Linear(embedding_dim, num_rhythm_classes)
        
    def forward(self, x: dict) -> torch.Tensor:
        """
        Args:
            x: Dictionary containing:
                - activations: [batch, bars, time, activation_dim]
                - features: [batch, bars, time, feature_dim]
                - bar_num: [batch, bars] tensor of bar numbers
        
        Returns:
            torch.Tensor: Logits of shape [batch, bars, time, num_classes]
        """
        # Get feature embeddings
        embeddings, rhythm = self.feature_encoder(x)  # [batch, bars, time, embedding_dim]
        
        # Reshape for transformer (combine batch and bars dimensions)
        batch_size, num_bars, seq_len, _ = embeddings.shape
        embeddings = embeddings.view(batch_size * num_bars, seq_len, -1)  # [batch*bars, time, embedding_dim]

        batch_size, num_bars, rhythm_seq_len, _ = rhythm.shape
        rhythm = rhythm.view(batch_size * num_bars, rhythm_seq_len, -1)  # [batch*bars, time, embedding_dim]
        
        # Concatenate embeddings and rhythm
        seq_len = embeddings.shape[-2]
        encoded_sequence = torch.cat([embeddings, rhythm], dim=-2)  # [batch*bars, time + beats, embedding_dim]

        # Create attention mask (all positions can attend to all other positions)
        mask = None
        
        # Apply transformer
        encoded = self.transformer(encoded_sequence, mask)  # [batch*bars, time, embedding_dim]
        
        # Split logits into bin and rhythm
        bin_logits = self.bin_output_projection(encoded[:, :seq_len, :])  # [batch*bars, time, num_classes]
        rhythm_logits = self.rhythm_output_projection(encoded[:, seq_len:, :])  # [batch*bars, time, num_classes]
        
        # Reshape back to original dimensions
        bin_logits = bin_logits.view(batch_size, num_bars, seq_len, -1)  # [batch, bars, time, num_classes]
        rhythm_logits = rhythm_logits.view(batch_size, num_bars, rhythm_seq_len, -1)  # [batch, bars, time, num_classes]
        
        return bin_logits, rhythm_logits
    
    def predict(self, x: dict) -> torch.Tensor:
        """
        Args:
            x: Same as forward()
            
        Returns:
            torch.Tensor: Predicted class indices of shape [batch, bars, time]
        """
        bin_logits, rhythm_logits = self.forward(x)
        return torch.argmax(bin_logits, dim=-1), torch.argmax(rhythm_logits, dim=-1) 

class TranscriptionMetrics:

    def _tokens_to_pianoroll(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert token predictions to pianoroll format.
        
        Args:
            tokens: Token indices of shape [batch, bars, time]
            
        Returns:
            torch.Tensor: Pianoroll of shape [batch, bars, time, 128] where each bin is 1 if a note is active
        """
        batch_size, num_bars, seq_len = tokens.shape
        pianoroll = torch.zeros((batch_size, num_bars, seq_len, 128), device=tokens.device)
        
        # For each position where we have a note (not rest), set the corresponding pitch bin to 1
        for b in range(batch_size):
            memorized_pitch = None
            for bar in range(num_bars):
                for t in range(seq_len):
                    if tokens[b, bar, t] < 128: # onset token
                        memorized_pitch = tokens[b, bar, t]
                        pianoroll[b, bar, t, memorized_pitch] = 1
                    elif tokens[b, bar, t] == 129: # rest token
                        memorized_pitch = None
                    elif tokens[b, bar, t] == 128: # tie token
                        if memorized_pitch is not None:
                            pianoroll[b, bar, t, memorized_pitch] = 1
                        else:
                            pianoroll[b, bar, t, :] = -1
        
        return pianoroll

    def _compute_pianoroll_accuracy(self, pred_pianoroll: torch.Tensor, true_pianoroll: torch.Tensor) -> torch.Tensor:
        """
        Compute accuracy at the pianoroll level (hit/miss for each time bin).
        
        Args:
            pred_pianoroll: Predicted pianoroll of shape [batch, bars, time, 128]
            true_pianoroll: Ground truth pianoroll of shape [batch, bars, time, 128]
            
        Returns:
            torch.Tensor: Pianoroll accuracy (scalar)
        """
        # Compare predictions with ground truth
        correct = (pred_pianoroll == true_pianoroll).all(dim=-1)  # [batch, bars, time]
        return correct.float().mean()
    
    def _compute_voiced_bin_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute accuracy for voiced bin detection, where all pitch tokens (1-127) are considered equivalent.
        
        Args:
            predictions: Predicted token indices of shape [batch*bars*time]
            targets: Ground truth token indices of shape [batch*bars*time]
            
        Returns:
            torch.Tensor: Voiced bin detection accuracy (scalar)
        """
        # Convert predictions and targets to binary (voiced vs not voiced)
        pred_voiced = predictions != 129 # not rest
        target_voiced = targets != 129 # not rest
        
        # Compute accuracy
        return (pred_voiced == target_voiced).float().mean()
    
        
    def DEPR_tokens_to_rhythm_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert a sequence of tokens to 4 rhythm tokens (one per beat).
        Each beat is represented by a single rhythm token based on the pattern of notes in that beat.
        
        Args:
            tokens: Token indices of shape [batch, bars, time]
            
        Returns:
            torch.Tensor: Rhythm tokens of shape [batch, bars, 4]
        """
        batch_size, num_bars, seq_len = tokens.shape
        rhythm_tokens = torch.zeros((batch_size, num_bars, 4), device=tokens.device)
        
        # Process each beat (12 subdivisions per beat)
        for b in range(batch_size):
            for bar in range(num_bars):
                rhythm_tokens[b, bar] = self.rhythm_tokenizer.encode(tokens[b, bar])
        
        return rhythm_tokens
    
    def DEPR_compute_bare_rhythm_accuracy(self, pred_tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute rhythm class accuracy by comparing rhythm tokens.
        
        Args:
            pred_tokens: Predicted token indices of shape [batch, bars, time]
            targets: Ground truth token indices of shape [batch, bars, time]
            
        Returns:
            torch.Tensor: Rhythm accuracy (scalar)
        """
        # Convert to rhythm tokens
        pred_rhythm = self._tokens_to_rhythm_tokens(pred_tokens).view(-1) 
        
        # Compare rhythm tokens
        correct = (pred_rhythm == targets).float()
        return correct.mean()

    def _compute_onset_recall(self, pred_tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_onsets = pred_tokens < 128
        true_onsets = targets < 128
        true_positives = (pred_onsets & true_onsets).float().sum()
        true_onsets_total = true_onsets.float().sum()
        return true_positives / (true_onsets_total + 1e-8)  # Add small epsilon to prevent division by zero

    def _compute_onset_precision(self, pred_tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_onsets = pred_tokens < 128
        true_onsets = targets < 128
        true_positives = (pred_onsets & true_onsets).float().sum()
        pred_onsets_total = pred_onsets.float().sum()
        return true_positives / (pred_onsets_total + 1e-8)  # Add small epsilon to prevent division by zero

    def _compute_onset_f1(self, pred_tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        precision = self._compute_onset_precision(pred_tokens, targets)
        recall = self._compute_onset_recall(pred_tokens, targets)
        return 2 * precision * recall / (precision + recall + 1e-8)  # Add small epsilon to prevent division by zero

    def _compute_special_pitch_accuracy(self, pred_tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_onsets = pred_tokens < 128
        true_onsets = targets < 128
        tp_onsets = (pred_onsets & true_onsets)
        # Only compare pitches where both prediction and target have onsets
        pitch_matches = (pred_tokens[tp_onsets] == targets[tp_onsets]).float()
        return pitch_matches.mean()
        
    
class RhythmScaffoldLightningModule(pl.LightningModule, TranscriptionMetrics):
    def __init__(self,
                 embedding_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 project_name: str = "solo-transcriber",
                 rhythm_loss_weight: float = 1.0,
                 teacher_forcing: bool = False):
        """
        Args:
            embedding_dim: Dimension of the embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            project_name: Name of the wandb project
        """
        super().__init__()
        self.teacher_forcing = teacher_forcing
        #self.rhythm_tokenizer = RhythmTokenizer()
        self.save_hyperparameters()
        
        # Create model
        self.model = JointPitchRhythmTransformerEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_bin_classes=128,
            num_rhythm_classes=44,
        )

        # Loss function
        self.bin_criterion = nn.CrossEntropyLoss()
        self.rhythm_criterion = nn.CrossEntropyLoss()
        
        # Initialize wandb
        self.wandb_logger = WandbLogger(
            project=project_name,
            log_model=True
        )
        
        # Log model architecture
        self.wandb_logger.watch(self.model, log="all", log_freq=100)

        self.rhythm_loss_weight = rhythm_loss_weight
        
    def _rhythm_instructed_loss(self, bin_logits: torch.Tensor, bin_targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bin_logits: [batch*bars*time, num_classes]
            bin_targets: [batch*bars*time]
            mask: [batch*bars*time]
        """
        indices = torch.where(mask == 0) # mask is 0 for note onsets
        masked_bin_logits = bin_logits.view(-1, bin_logits.shape[-1])[indices]  # [N, num_classes]
        masked_bin_targets = bin_targets.view(-1)[indices]  # [N]
        
        # Ensure logits are float and targets are long
        masked_bin_logits = masked_bin_logits.float()
        masked_bin_targets = masked_bin_targets.long()
        
        # Handle empty tensor case
        if masked_bin_logits.numel() == 0:
            return torch.tensor(0.0, device=bin_logits.device)
            
        return self.bin_criterion(masked_bin_logits, masked_bin_targets - 1)
    
    def _build_scaffold(self, code: str, device: torch.device):
        """
        Replace all pitches with 0, rest with 129, and tie with 128.
        """
        scaffold = []
        for ch in code:
            if ch == 'r':
                scaffold.append(129) # rest
            elif ch == 't':
                scaffold.append(128) # tie
            elif ch == 'o':
                scaffold.append(0)
        return torch.tensor(scaffold, device=device)

    def _generate_structured_predictions(self, bin_logits: torch.Tensor, rhythm_logits: torch.Tensor):
        filtered_rhythm_logits = rhythm_logits[:, :-2] # not using <rare> and <too fast> tokens for prediction 
        rhythm_predictions = torch.argmax(filtered_rhythm_logits, dim=-1)
        scaffolds = [self._build_scaffold(INV_RHYTHM_TOKENS[int(code)], bin_logits.device) for code in rhythm_predictions.view(-1)]
        scaffolds = torch.stack(scaffolds, dim=0).view(-1)
        indices = torch.where(scaffolds == 0) # 0 is the masked pitch token
        bin_predictions = torch.argmax(bin_logits, dim=-1) + 1
        final = scaffolds.clone()
        final[indices] = bin_predictions[indices]
        return final

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)
    
    def generic_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int, mode: str) -> torch.Tensor:
        x, y = batch['x'], batch['y']
        bin_logits, rhythm_logits = self(x)
         
        # Reshape for loss computation
        batch_size, num_bars, seq_len, num_classes = bin_logits.shape
        _, _, rhythm_seq_len, num_rhythm_classes = rhythm_logits.shape
        bin_logits = bin_logits.view(-1, num_classes)  # [batch*bars*time, num_classes]
        rhythm_logits = rhythm_logits.view(-1, num_rhythm_classes)  # [batch*bars*time, num_classes]
        targets = y['tokens'].view(-1)  # [batch*bars*time]
        rhythm_targets = y['rhythm_tokens'].view(-1)  # [batch*bars*time]
        mask = y['mask'].view(-1) # [batch*bars*time]
        
        # Compute loss
        if self.teacher_forcing:
            loss_pitch = self._rhythm_instructed_loss(bin_logits, targets, mask) 
            loss_rhythm = self.rhythm_criterion(rhythm_logits, rhythm_targets)
            loss = loss_pitch + self.rhythm_loss_weight * loss_rhythm
        else:
            raise NotImplementedError("don't know what to do here -- without teacher forcing we don't have ground truth")
        
        structured_predictions = self._generate_structured_predictions(bin_logits, rhythm_logits)

        # Compute token-level accuracy
        token_accuracy = (structured_predictions == targets).float().mean()
        
        # Compute voiced bin detection accuracy
        voiced_bin_accuracy = self._compute_voiced_bin_accuracy(structured_predictions, targets)

        # Compute onset precision, recall, and f1
        onset_precision = self._compute_onset_precision(structured_predictions, targets)
        onset_recall = self._compute_onset_recall(structured_predictions, targets)
        onset_f1 = self._compute_onset_f1(structured_predictions, targets)
        special_pitch_accuracy = self._compute_special_pitch_accuracy(structured_predictions, targets)
        
        # Compute rhythm accuracy from bin predictions
        pred_tokens = structured_predictions.view(batch_size, num_bars, seq_len)
        true_tokens = targets.view(batch_size, num_bars, seq_len)
        #rhythm_accuracy = self._compute_bare_rhythm_accuracy(pred_tokens, rhythm_targets)
        
        # Compute pianoroll-level accuracy
        pred_pianoroll = self._tokens_to_pianoroll(pred_tokens)
        true_pianoroll = self._tokens_to_pianoroll(true_tokens)
        pianoroll_accuracy = self._compute_pianoroll_accuracy(pred_pianoroll, true_pianoroll)
        
        # Compute rhythm accuracy from rhythm predictions
        rhythm_predictions = torch.argmax(rhythm_logits, dim=-1)
        pred_rhythm_tokens = rhythm_predictions.view(batch_size, num_bars, rhythm_seq_len)
        true_rhythm_tokens = rhythm_targets.view(batch_size, num_bars, rhythm_seq_len)
        rhythm_accuracy_from_rhythm_predictions = (pred_rhythm_tokens == true_rhythm_tokens).float().mean()

        # Log metrics
        if mode == 'train':
            onstep = True
        else:
            onstep = False
        self.log(f'{mode}_loss', loss, on_step=onstep, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_loss_pitch', loss_pitch, on_step=onstep, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_loss_rhythm', loss_rhythm, on_step=onstep, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_token_accuracy', token_accuracy, on_step=onstep, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_voiced_bin_accuracy', voiced_bin_accuracy, on_step=onstep, on_epoch=True, prog_bar=True)
        #self.log(f'{mode}_rhythm_accuracy', rhythm_accuracy, on_step=onstep, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_pianoroll_accuracy', pianoroll_accuracy, on_step=onstep, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_rhythm_accuracy_new', rhythm_accuracy_from_rhythm_predictions, on_step=onstep, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_onset_precision', onset_precision, on_step=onstep, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_onset_recall', onset_recall, on_step=onstep, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_onset_f1', onset_f1, on_step=onstep, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_special_pitch_accuracy', special_pitch_accuracy, on_step=onstep, on_epoch=True, prog_bar=True)

        # Log learning rate
        # self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])
        
        
        return loss


    def training_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self.generic_step(batch, batch_idx, 'train')
        
    def validation_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self.generic_step(batch, batch_idx, 'val')
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def predict_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch['x']
        bin_logits, rhythm_logits = self(x)
        return torch.argmax(bin_logits, dim=-1), torch.argmax(rhythm_logits, dim=-1)
    
    @staticmethod
    def get_trainer_callbacks():
        """Get callbacks for the trainer"""
        return [] 
