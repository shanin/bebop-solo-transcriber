import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from typing import Dict, Any, Optional, List, Tuple
import json
from datetime import datetime
from src.tokenizer.rhythm_tokens import RHYTHM_TOKENS
INV_RHYTHM_TOKENS = {v['id']: k for k, v in RHYTHM_TOKENS.items()}

class PositionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 128, fourier_dim: int = 13):
        """
        Args:
            embedding_dim: Dimension of the output embeddings
            fourier_dim: Dimension of the fourier features for inter-beat position
        """
        super().__init__()
        
        # Beat embeddings (4 beats per bar)
        self.beat_embedding = nn.Embedding(4, embedding_dim)
        
        # Subdivision embeddings (12 subdivisions per beat + 1 for rhythm token)
        self.subdivision_embedding = nn.Embedding(13, embedding_dim)
        
        # Bar embeddings (for different bar positions in the sequence)
        self.bar_embedding = nn.Embedding(32, embedding_dim)  # Assuming max 32 bars per sequence

        # MLP for processing fourier features for frame-level input
        self.fourier_mlp = nn.Sequential(
            nn.Linear(fourier_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward_latent_encoding(self, bin_position_encoding: torch.Tensor, beat_position_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode position for latent variables (bin and rhythm latents).
        
        Args:
            bin_position_encoding: [batch, bars, 48, 3] - [bar_idx, beat_idx, bin_idx]
            beat_position_encoding: [batch, bars, 4, 3] - [bar_idx, beat_idx, 0]
                
        Returns:
            Tuple of (bin_position_emb, rhythm_position_emb)
        """
        # Extract position indices
        bin_bar_indices = bin_position_encoding[:, :, :, 0]  # [batch, bars, 48]
        bin_beat_indices = bin_position_encoding[:, :, :, 1]  # [batch, bars, 48]
        bin_subdivision_indices = bin_position_encoding[:, :, :, 2]  # [batch, bars, 48]
        
        rhythm_bar_indices = beat_position_encoding[:, :, :, 0]  # [batch, bars, 4]
        rhythm_beat_indices = beat_position_encoding[:, :, :, 1]  # [batch, bars, 4]
        
        # Convert to long and add debug assertions for latent position encoding
        bin_bar_indices_long = bin_bar_indices.long()
        bin_beat_indices_long = bin_beat_indices.long()
        bin_subdivision_indices_long = bin_subdivision_indices.long()
        rhythm_bar_indices_long = rhythm_bar_indices.long()
        rhythm_beat_indices_long = rhythm_beat_indices.long()
        
        # Debug assertions for bin latent indices
        assert bin_bar_indices_long.min() >= 0, f"bin_bar_indices contains negative values: min={bin_bar_indices_long.min()}"
        assert bin_bar_indices_long.max() < 32, f"bin_bar_indices contains values >= 32: max={bin_bar_indices_long.max()}"
        assert bin_beat_indices_long.min() >= 0, f"bin_beat_indices contains negative values: min={bin_beat_indices_long.min()}"
        assert bin_beat_indices_long.max() < 4, f"bin_beat_indices contains values >= 4: max={bin_beat_indices_long.max()}"
        assert bin_subdivision_indices_long.min() >= 0, f"bin_subdivision_indices contains negative values: min={bin_subdivision_indices_long.min()}"
        assert bin_subdivision_indices_long.max() < 13, f"bin_subdivision_indices contains values >= 13: max={bin_subdivision_indices_long.max()}"
        
        # Debug assertions for rhythm latent indices  
        assert rhythm_bar_indices_long.min() >= 0, f"rhythm_bar_indices contains negative values: min={rhythm_bar_indices_long.min()}"
        assert rhythm_bar_indices_long.max() < 32, f"rhythm_bar_indices contains values >= 32: max={rhythm_bar_indices_long.max()}"
        assert rhythm_beat_indices_long.min() >= 0, f"rhythm_beat_indices contains negative values: min={rhythm_beat_indices_long.min()}"
        assert rhythm_beat_indices_long.max() < 4, f"rhythm_beat_indices contains values >= 4: max={rhythm_beat_indices_long.max()}"
        
        # Get embeddings for bin latents
        bin_bar_emb = self.bar_embedding(bin_bar_indices_long)
        bin_beat_emb = self.beat_embedding(bin_beat_indices_long)
        bin_subdivision_emb = self.subdivision_embedding(bin_subdivision_indices_long)
        bin_position_emb = bin_bar_emb + bin_beat_emb + bin_subdivision_emb
        
        # Get embeddings for rhythm latents
        rhythm_bar_emb = self.bar_embedding(rhythm_bar_indices_long)
        rhythm_beat_emb = self.beat_embedding(rhythm_beat_indices_long)
        rhythm_subdivision_emb = self.subdivision_embedding(torch.full_like(rhythm_beat_indices, 12).long())  # Use 12 for rhythm token
        rhythm_position_emb = rhythm_bar_emb + rhythm_beat_emb + rhythm_subdivision_emb
        
        return bin_position_emb, rhythm_position_emb
    
    def forward_frame_encoding(self, frame_posenc: torch.Tensor) -> torch.Tensor:
        """
        Encode position for frame-level input from CRNN frontend.
        
        Args:
            frame_posenc: [batch, frames, posenc_dim] where:
                - posenc[:, :, 0] = bar index
                - posenc[:, :, 1] = beat index (0-3)
                - posenc[:, :, 2:] = fourier features for inter-beat position
                
        Returns:
            torch.Tensor: Frame position embeddings [batch, frames, embedding_dim]
        """
        # Extract position indices and fourier features
        bar_indices = frame_posenc[:, :, 0].long()  # [batch, frames]
        beat_indices = frame_posenc[:, :, 1].long()  # [batch, frames]
        fourier_features = frame_posenc[:, :, 2:]  # [batch, frames, fourier_dim]
        
        # Debug assertions for position encoding data
        assert frame_posenc.dtype == torch.float32, f"frame_posenc has wrong dtype: {frame_posenc.dtype}, expected float32"
        assert bar_indices.min() >= 0, f"bar_indices contains negative values: min={bar_indices.min()}"
        assert bar_indices.max() < 32, f"bar_indices contains values >= 32: max={bar_indices.max()}"
        assert beat_indices.min() >= 0, f"beat_indices contains negative values: min={beat_indices.min()}"
        assert beat_indices.max() < 4, f"beat_indices contains values >= 4: max={beat_indices.max()}"
        
        # Get embeddings
        bar_emb = self.bar_embedding(bar_indices)  # [batch, frames, embedding_dim]
        beat_emb = self.beat_embedding(beat_indices)  # [batch, frames, embedding_dim]
        fourier_emb = self.fourier_mlp(fourier_features)  # [batch, frames, embedding_dim]
        
        # Sum the embeddings
        frame_position_emb = bar_emb + beat_emb + fourier_emb  # [batch, frames, embedding_dim]
        
        return frame_position_emb


class PerceiverCrossAttention(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_heads: int = 8, dropout: float = 0.1):
        """
        Cross-attention layer for Perceiver-style processing.
        Latents emit queries, frames emit keys and values.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latents: torch.Tensor, frames: torch.Tensor, frame_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            latents: [batch, num_latents, embedding_dim] - queries
            frames: [batch, num_frames, embedding_dim] - keys and values
            frame_mask: [batch, num_frames] - mask for valid frames (True = valid, False = padded)
            
        Returns:
            torch.Tensor: Updated latents [batch, num_latents, embedding_dim]
        """
        # Convert mask format for attention (True = ignore, False = attend)
        if frame_mask is not None:
            # Invert mask: frame_mask has True for valid, attention needs False for valid
            attention_mask = ~frame_mask  # [batch, num_frames]
        else:
            attention_mask = None
            
        # Cross-attention: latents query, frames provide keys and values
        attended_latents, _ = self.cross_attention(
            query=latents,
            key=frames,
            value=frames,
            key_padding_mask=attention_mask
        )
        
        # Residual connection and normalization
        latents = self.norm(latents + self.dropout(attended_latents))
        
        return latents


class CRNNFeatureEncoder(nn.Module):
    def __init__(self, 
                 frame_feature_dim: int = 88,  # CRNN output features (onsets, offsets, frames)
                 embedding_dim: int = 128,
                 fourier_dim: int = 13):
        """
        Feature encoder for CRNN frontend output.
        
        Args:
            frame_feature_dim: Dimension of frame-level features from CRNN (default: 88 for piano keys)
            embedding_dim: Dimension of the output embeddings
            fourier_dim: Dimension of fourier features in position encoding
        """
        super().__init__()
        
        # Frame feature encoders for CRNN outputs
        self.onset_encoder = nn.Sequential(
            nn.Linear(frame_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
        
        self.offset_encoder = nn.Sequential(
            nn.Linear(frame_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
        
        self.frame_encoder = nn.Sequential(
            nn.Linear(frame_feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

        # Fusion layer for combining frame features
        self.frame_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

        # Learnable latent variables
        self.bin_latents = nn.Parameter(torch.randn(1, 48, embedding_dim))  # 12 per beat * 4 beats = 48
        self.rhythm_latents = nn.Parameter(torch.randn(1, 4, embedding_dim))  # 1 per beat = 4
        
        # Position embedding
        self.position_embedding = PositionEmbedding(embedding_dim, fourier_dim)
        
        # Cross-attention layer
        self.cross_attention = PerceiverCrossAttention(embedding_dim)
        
    def forward(self, x: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Dictionary containing:
                - frame_level: Dict with 'onsets', 'offsets', 'frames', 'posenc'
                - frame_mask: [batch, frames] mask for valid frames
                - bin_position_encoding: [batch, bars, 48, 3]
                - beat_position_encoding: [batch, bars, 4, 3]
        
        Returns:
            Tuple of (bin_embeddings, rhythm_embeddings)
        """
        # Get frame-level data
        frame_data = x['frame_level']
        frame_mask = x['frame_mask']
        
        # Encode frame features
        onsets = frame_data['onsets']  # [batch, frames, 88]
        offsets = frame_data['offsets']  # [batch, frames, 88]
        frames = frame_data['frames']  # [batch, frames, 88]
        posenc = frame_data['posenc']  # [batch, frames, posenc_dim]
        
        batch_size, num_frames, _ = onsets.shape
        
        # Encode individual frame features
        onset_emb = self.onset_encoder(onsets)  # [batch, frames, embedding_dim]
        offset_emb = self.offset_encoder(offsets)  # [batch, frames, embedding_dim]
        frame_emb = self.frame_encoder(frames)  # [batch, frames, embedding_dim]
        
        # Fuse frame features
        fused_frames = torch.cat([onset_emb, offset_emb, frame_emb], dim=-1)  # [batch, frames, 3*embedding_dim]
        fused_frames = self.frame_fusion(fused_frames)  # [batch, frames, embedding_dim]
        
        # Add position encoding to frames
        frame_pos_emb = self.position_embedding.forward_frame_encoding(posenc)  # [batch, frames, embedding_dim]
        encoded_frames = fused_frames + frame_pos_emb  # [batch, frames, embedding_dim]
        
        # Prepare latents with position encoding
        bin_position_encoding = x['bin_position_encoding']  # [batch, bars, 48, 3]
        beat_position_encoding = x['beat_position_encoding']  # [batch, bars, 4, 3]
        
        # Get number of bars from position encoding
        num_bars = bin_position_encoding.shape[1]
        
        # Expand latents for batch and bars
        bin_latents = self.bin_latents.expand(batch_size, num_bars, -1, -1)  # [batch, bars, 48, embedding_dim]
        rhythm_latents = self.rhythm_latents.expand(batch_size, num_bars, -1, -1)  # [batch, bars, 4, embedding_dim]
        
        # Add position encoding to latents
        bin_pos_emb, rhythm_pos_emb = self.position_embedding.forward_latent_encoding(
            bin_position_encoding, beat_position_encoding
        )
        bin_latents = bin_latents + bin_pos_emb  # [batch, bars, 48, embedding_dim]
        rhythm_latents = rhythm_latents + rhythm_pos_emb  # [batch, bars, 4, embedding_dim]
        
        # Reshape for cross-attention (flatten bars dimension)
        bin_latents = bin_latents.view(batch_size, num_bars * 48, -1)  # [batch, bars*48, embedding_dim]
        rhythm_latents = rhythm_latents.view(batch_size, num_bars * 4, -1)  # [batch, bars*4, embedding_dim]
        
        # Combine latents for cross-attention
        all_latents = torch.cat([bin_latents, rhythm_latents], dim=1)  # [batch, bars*48 + bars*4, embedding_dim]
        
        # Cross-attention: latents attend to frames
        attended_latents = self.cross_attention(all_latents, encoded_frames, frame_mask)
        
        # Split back into bin and rhythm latents
        bin_attended = attended_latents[:, :num_bars * 48, :]  # [batch, bars*48, embedding_dim]
        rhythm_attended = attended_latents[:, num_bars * 48:, :]  # [batch, bars*4, embedding_dim]
        
        # Reshape back to original dimensions
        bin_attended = bin_attended.view(batch_size, num_bars, 48, -1)  # [batch, bars, 48, embedding_dim]
        rhythm_attended = rhythm_attended.view(batch_size, num_bars, 4, -1)  # [batch, bars, 4, embedding_dim]
        
        return bin_attended, rhythm_attended

    
class RhythmPerceiver(nn.Module):
    def __init__(self, 
                 embedding_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_backend_heads: int = 8,
                 num_backend_layers: int = 2,
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
        
        # Feature encoder for CRNN frontend
        self.feature_encoder = CRNNFeatureEncoder(embedding_dim=embedding_dim)
 
        
        # Transformer encoder layers for joint processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Additional transformer layers for bin processing after structural injection
        bin_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_backend_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.bin_transformer = nn.TransformerEncoder(bin_encoder_layer, num_layers=num_backend_layers)
        
        # Output projection
        self.bin_output_projection = nn.Linear(embedding_dim, num_bin_classes)
        self.rhythm_output_projection = nn.Linear(embedding_dim, num_rhythm_classes)

        self.onset_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.rest_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.tie_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.special_token = nn.Parameter(torch.randn(1, 1, embedding_dim))  # for 'x' (too fast)
        
        # Precompute scaffold indices for each rhythm token
        # Each rhythm token corresponds to a 12-step pattern (one beat)
        scaffold_indices = self._build_scaffold_lookup_table()
        self.register_buffer('scaffold_indices', scaffold_indices)
        
    def _build_scaffold_lookup_table(self) -> torch.Tensor:
        """
        Build a lookup table of scaffold indices for each rhythm token.
        
        Returns:
            torch.Tensor: Shape [num_rhythm_tokens, 12] where values are:
                - 0 = onset position (use onset_token)
                - 1 = rest position (use rest_token) 
                - 2 = tie position (use tie_token)
        """
        scaffold_indices = []
        
        for rhythm_id in range(len(INV_RHYTHM_TOKENS)):
            rhythm_code = INV_RHYTHM_TOKENS[rhythm_id]
            indices = []
            
            for ch in rhythm_code:
                if ch == 'r':
                    indices.append(1)  # rest -> index 1
                elif ch == 't':
                    indices.append(2)  # tie -> index 2
                elif ch == 'o':
                    indices.append(0)  # onset -> index 0
            
            scaffold_indices.append(indices)
        
        return torch.tensor(scaffold_indices, dtype=torch.long)
        
    def forward(self, x: dict, injected_mask: torch.Tensor = None, use_teacher_forcing: bool = False) -> torch.Tensor:
        """
        Args:
            x: Dictionary containing:
                - frame_level: Dict with 'onsets', 'offsets', 'frames', 'posenc'
                - frame_mask: [batch, frames] mask for valid frames
                - bin_position_encoding: [batch, bars, 48, 3]
                - beat_position_encoding: [batch, bars, 4, 3]

            injected_mask: [batch, bars, bins] tensor of mask for structural injection. 
            if None, no structural injection is performed - mask is build from predicted rhythm tokens
        
        Returns:
            Tuple of (bin_logits, rhythm_logits)
        """
        # Get feature embeddings from CRNN cross-attention
        bin_embeddings, rhythm_embeddings = self.feature_encoder(x)  # [batch, bars, 48, embedding_dim], [batch, bars, 4, embedding_dim]
        
        # Reshape for transformer processing
        batch_size, num_bars, bin_seq_len, _ = bin_embeddings.shape  # bin_seq_len = 48
        batch_size, num_bars, rhythm_seq_len, _ = rhythm_embeddings.shape  # rhythm_seq_len = 4
        
        embeddings = bin_embeddings.view(batch_size, num_bars * bin_seq_len, -1)  # [batch, bars*48, embedding_dim]
        rhythm = rhythm_embeddings.view(batch_size, num_bars * rhythm_seq_len, -1)  # [batch, bars*4, embedding_dim]
        
        # Concatenate embeddings and rhythm along time dimension
        encoded_sequence = torch.cat([embeddings, rhythm], dim=-2)  # [batch, bars*time + bars*beats, embedding_dim]

        # Create attention mask (all positions can attend to all other positions)
        mask = None
        
        # Apply transformer
        encoded = self.transformer(encoded_sequence, mask)  # [batch, bars*time + bars*beats, embedding_dim]
        
        # Split encoded sequence back into bin and rhythm parts
        bin_encoded = encoded[:, :embeddings.shape[1], :]  # [batch, bars*time, embedding_dim]
        rhythm_encoded = encoded[:, embeddings.shape[1]:, :]  # [batch, bars*beats, embedding_dim]
        
        # Get rhythm logits and predictions
        rhythm_logits = self.rhythm_output_projection(rhythm_encoded)  # [batch, bars*beats, num_rhythm_classes]
        
        if use_teacher_forcing and injected_mask is not None:
            mask_flat = injected_mask.view(batch_size, -1)  # [batch, bars*bins]
            
            # Ensure mask has correct length for 48 bins per bar
            expected_length = batch_size * num_bars * bin_seq_len  # bin_seq_len = 48
            assert mask_flat.numel() == expected_length, f"injected_mask has wrong size: {mask_flat.numel()} vs expected {expected_length}"
            assert mask_flat.max() <= 3, f"injected_mask contains values > 3: max={mask_flat.max()}"
            assert mask_flat.min() >= 0, f"injected_mask contains values < 0: min={mask_flat.min()}"
            
            scaffold_indices = mask_flat  # Use directly if already in [0, 1, 2, 3] format
        else:

            # Filter out rare and too fast tokens for prediction
            filtered_rhythm_logits = rhythm_logits[:, :, :-2]  # [batch, bars*beats, num_rhythm_classes-2]
            rhythm_predictions = torch.argmax(filtered_rhythm_logits, dim=-1)  # [batch, bars*beats]
            
            # Get scaffold indices for each rhythm prediction
            # rhythm_predictions: [batch, bars*beats] where beats = rhythm_seq_len = 4
            # self.scaffold_indices: [num_rhythm_tokens, 12]
            scaffold_per_beat = self.scaffold_indices[rhythm_predictions]  # [batch, bars*4, 12]
            
            # Reshape to match bin sequence length (48 bins per bar)
            # Each bar has 4 beats, each beat has 12 bins
            scaffold_indices = scaffold_per_beat.view(batch_size, -1)  # [batch, bars*beats*12]
            #scaffold_indices = scaffold_indices.view(batch_size, num_bars, 48)  # [batch, bars, 48] 
            #scaffold_indices = scaffold_indices.view(batch_size, -1)  # [batch, bars*48]
        
        # Create structural token embeddings using learned parameters
        # Stack the four learned tokens: [onset, rest, tie, special]
        structural_tokens = torch.stack([
            self.onset_token.squeeze(),  # [embedding_dim]
            self.rest_token.squeeze(),   # [embedding_dim] 
            self.tie_token.squeeze(),    # [embedding_dim]
            self.special_token.squeeze() # [embedding_dim]
        ], dim=0)  # [4, embedding_dim]
        
        # Use embedding lookup to get structural embeddings
        # scaffold_indices: [batch, bars*48] with values 0, 1, 2, 3
        # structural_tokens: [4, embedding_dim]
        structural_embeddings = F.embedding(scaffold_indices, structural_tokens)  # [batch, bars*48, embedding_dim]
        
        # Inject structural information into bin embeddings
        # bin_encoded comes from main transformer and should have shape [batch, bars*48, embedding_dim]
        assert bin_encoded.shape[1] == num_bars * bin_seq_len, f"bin_encoded has wrong shape: {bin_encoded.shape} vs expected [batch, {num_bars * bin_seq_len}, embedding_dim]"
        bin_encoded_with_structure = bin_encoded + structural_embeddings  # [batch, bars*48, embedding_dim]
        
        # Apply additional transformer layers to bin embeddings with structural information
        bin_encoded_final = self.bin_transformer(bin_encoded_with_structure, mask)  # [batch, bars*48, embedding_dim]
        
        # Get final bin logits
        bin_logits = self.bin_output_projection(bin_encoded_final)  # [batch, bars*48, num_classes]
        
        # Reshape back to original dimensions
        bin_logits = bin_logits.view(batch_size, num_bars, bin_seq_len, -1)  # [batch, bars, 48, num_classes]
        rhythm_logits = rhythm_logits.view(batch_size, num_bars, rhythm_seq_len, -1)  # [batch, bars, 4, num_classes]
        
        return bin_logits, rhythm_logits
    
    def predict(self, x: dict, injected_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Same as forward()
            injected_mask: Mask for structural injection (optional for inference)
            
        Returns:
            torch.Tensor: Predicted class indices of shape [batch, bars, time]
        """
        bin_logits, rhythm_logits = self.forward(x, injected_mask, use_teacher_forcing=False)
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
        
    
class RhythmPerceiverLightningModule(pl.LightningModule, TranscriptionMetrics):
    def __init__(self,
                 embedding_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_backend_heads: int = 8,
                 num_backend_layers: int = 2,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 project_name: str = "solo-transcriber",
                 experiment_name: str = "default",
                 rhythm_loss_weight: float = 1.0
    ):
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
        self.save_hyperparameters()
        
        # Create model
        self.model = RhythmPerceiver(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_backend_heads=num_backend_heads,
            num_backend_layers=num_backend_layers,
            dropout=dropout,
            num_bin_classes=128,  # 128 MIDI pitches (0-127)
            num_rhythm_classes=44,
        )

        # Loss function with label smoothing to reduce overfitting
        self.bin_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.rhythm_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Initialize wandb
        if experiment_name == "default":
            name = f'{project_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        else:
            name = experiment_name
        self.wandb_logger = WandbLogger(
            project=project_name,
            name=name,
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
            
        return self.bin_criterion(masked_bin_logits, masked_bin_targets)
    
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

    def forward(self, x: Dict[str, torch.Tensor], injected_mask: torch.Tensor = None, use_teacher_forcing: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, injected_mask, use_teacher_forcing)
    
    def generic_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int, mode: str) -> torch.Tensor:
        # Extract data from new backend dataset structure
        bin_level = batch['bin_level']
        frame_level = batch['frame_level']
        frame_mask = batch['frame_mask']
        bin_position_encoding = batch['bin_position_encoding']
        beat_position_encoding = batch['beat_position_encoding']
        meta = batch['meta']
        
        # Prepare input dictionary for model
        x = {
            'frame_level': frame_level,
            'frame_mask': frame_mask,
            'bin_position_encoding': bin_position_encoding,
            'beat_position_encoding': beat_position_encoding
        }
        
        # Get targets from bin_level data
        injected_mask = bin_level['mask']
        bin_logits, rhythm_logits = self(x, injected_mask, use_teacher_forcing=True)
         
        # Reshape for loss computation
        batch_size, num_bars, seq_len, num_classes = bin_logits.shape
        _, _, rhythm_seq_len, num_rhythm_classes = rhythm_logits.shape
        bin_logits = bin_logits.view(-1, num_classes)  # [batch*bars*time, num_classes]
        rhythm_logits = rhythm_logits.view(-1, num_rhythm_classes)  # [batch*bars*time, num_classes]
        targets = bin_level['tokens'].view(-1)  # [batch*bars*bins]
        rhythm_targets = bin_level['rhythm_tokens'].view(-1)  # [batch*bars*beats]
        mask = bin_level['mask'].view(-1) # [batch*bars*bins]
        
        # Compute loss
        loss_pitch = self._rhythm_instructed_loss(bin_logits, targets, mask) 
        loss_rhythm = self.rhythm_criterion(rhythm_logits, rhythm_targets)
        loss = loss_pitch + self.rhythm_loss_weight * loss_rhythm
                
        # Log differently for train vs val to reduce noise
        if mode == 'train':
            # For training: only log per-epoch to reduce console noise
            self.log(f'{mode}_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f'{mode}_loss_pitch', loss_pitch, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f'{mode}_loss_rhythm', loss_rhythm, on_step=False, on_epoch=True, prog_bar=False)
        else:
            # For validation: log both step and epoch
            self.log(f'{mode}_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f'{mode}_loss_pitch', loss_pitch, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f'{mode}_loss_rhythm', loss_rhythm, on_step=False, on_epoch=True, prog_bar=False)

        return loss


    def training_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self.generic_step(batch, batch_idx, 'train')
        
    def validation_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self.generic_step(batch, batch_idx, 'val')
    
    def test_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:
        mode = 'test'
        # Extract data from new backend dataset structure
        bin_level = batch['bin_level']
        frame_level = batch['frame_level']
        frame_mask = batch['frame_mask']
        bin_position_encoding = batch['bin_position_encoding']
        beat_position_encoding = batch['beat_position_encoding']
        meta = batch['meta']
        
        # Prepare input dictionary for model
        x = {
            'frame_level': frame_level,
            'frame_mask': frame_mask,
            'bin_position_encoding': bin_position_encoding,
            'beat_position_encoding': beat_position_encoding
        }
        
        # Get predictions
        bin_logits, rhythm_logits = self(x, injected_mask = None, use_teacher_forcing=False)
         
        # Reshape for loss computation
        batch_size, num_bars, seq_len, num_classes = bin_logits.shape
        _, _, rhythm_seq_len, num_rhythm_classes = rhythm_logits.shape
        bin_logits = bin_logits.view(-1, num_classes)  # [batch*bars*time, num_classes]
        rhythm_logits = rhythm_logits.view(-1, num_rhythm_classes)  # [batch*bars*time, num_classes]
        targets = bin_level['tokens'].view(-1)  # [batch*bars*bins]
        rhythm_targets = bin_level['rhythm_tokens'].view(-1)  # [batch*bars*beats]
        mask = bin_level['mask'].view(-1) # [batch*bars*bins]
        
        # Generate structured predictions
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

        # Log test metrics - only show most important ones in progress bar
        self.log(f'{mode}_token_accuracy', token_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_voiced_bin_accuracy', voiced_bin_accuracy, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{mode}_pianoroll_accuracy', pianoroll_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_rhythm_accuracy_new', rhythm_accuracy_from_rhythm_predictions, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{mode}_onset_precision', onset_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{mode}_onset_recall', onset_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{mode}_onset_f1', onset_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_special_pitch_accuracy', special_pitch_accuracy, on_step=False, on_epoch=True, prog_bar=False)

        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Add learning rate scheduler to reduce overfitting
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def predict_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract data from new backend dataset structure
        frame_level = batch['frame_level']
        frame_mask = batch['frame_mask']
        bin_position_encoding = batch['bin_position_encoding']
        beat_position_encoding = batch['beat_position_encoding']
        
        # Prepare input dictionary for model
        x = {
            'frame_level': frame_level,
            'frame_mask': frame_mask,
            'bin_position_encoding': bin_position_encoding,
            'beat_position_encoding': beat_position_encoding
        }
        
        injected_mask = None  # Use predicted rhythm during inference
        bin_logits, rhythm_logits = self(x, injected_mask, use_teacher_forcing=False)
        return torch.argmax(bin_logits, dim=-1), torch.argmax(rhythm_logits, dim=-1)
    
    @staticmethod
    def get_trainer_callbacks():
        """Get callbacks for the trainer"""
        return [] 
    
    def create_trainer_with_wandb(self, 
                                 max_epochs: int = 100,
                                 accelerator: str = "auto",
                                 devices: str = "auto",
                                 precision: str = "32",
                                 check_val_every_n_epoch: int = 1,
                                 log_every_n_steps: int = 50,
                                 enable_checkpointing: bool = True,
                                 checkpoint_monitor: str = "val_loss",
                                 checkpoint_mode: str = "min",
                                 checkpoint_save_top_k: int = 3,
                                 **trainer_kwargs) -> Trainer:
        """
        Create a PyTorch Lightning trainer configured with wandb logging and model checkpointing.
        
        Args:
            max_epochs: Maximum number of training epochs
            accelerator: Training accelerator (auto, cpu, gpu, etc.)
            devices: Number of devices to use (auto, 1, 2, etc.)
            precision: Training precision (16-mixed, 32, etc.)
            check_val_every_n_epoch: Validation frequency
            log_every_n_steps: Logging frequency
            enable_checkpointing: Whether to enable model checkpointing
            checkpoint_monitor: Metric to monitor for checkpointing
            checkpoint_mode: Mode for checkpointing (min or max)
            checkpoint_save_top_k: Number of best checkpoints to save
            **trainer_kwargs: Additional arguments to pass to the trainer
            
        Returns:
            Configured PyTorch Lightning Trainer
        """
        # Set up callbacks
        callbacks = self.get_trainer_callbacks()
        
        # Add model checkpoint callback if checkpointing is enabled
        if enable_checkpointing:
            checkpoint_callback = ModelCheckpoint(
                monitor=checkpoint_monitor,
                mode=checkpoint_mode,
                save_top_k=checkpoint_save_top_k,
                save_last=True,
                filename=f'{self.hparams.experiment_name}-{{epoch:02d}}-{{{checkpoint_monitor}:.4f}}',
                auto_insert_metric_name=False,
            )
            callbacks.append(checkpoint_callback)
        
        # Create trainer with wandb logger
        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            logger=self.wandb_logger,
            callbacks=callbacks,
            check_val_every_n_epoch=check_val_every_n_epoch,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=enable_checkpointing,
            **trainer_kwargs
        )
        
        return trainer 
