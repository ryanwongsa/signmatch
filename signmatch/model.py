from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

from .hiera import mae_hiera_base_16x224

BACKBONE_DIM = 768
EMBED_DIM = 512
PATCH_KEY = "backbone.patch_embed.proj.weight"


class DictionaryProjection(nn.Module):
    def __init__(self, input_size=BACKBONE_DIM, hidden_size=EMBED_DIM, out_dim=EMBED_DIM):
        super().__init__()
        self.lin = nn.Linear(input_size, hidden_size)
        self.projection_dict = nn.LSTM(
            hidden_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True
        )
        self.proj = nn.Linear(2 * hidden_size, out_dim)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.lin(feats)
        packed = pack_sequence(torch.split(x, lengths.tolist()), enforce_sorted=False)
        _, (h_n, _) = self.projection_dict(packed)
        z = self.proj(torch.cat((h_n[-2], h_n[-1]), dim=1))
        return F.normalize(z, dim=-1)


class SignMatch(nn.Module):
    def __init__(self, temporal_kernel: int = 5):
        super().__init__()
        self.backbone = mae_hiera_base_16x224(num_classes=400, pretrained=False)
        old = self.backbone.patch_embed.proj
        self.backbone.patch_embed.proj = nn.Conv3d(
            old.in_channels,
            old.out_channels,
            kernel_size=(temporal_kernel, 7, 7),
            stride=(4, 4, 4),
            padding=(temporal_kernel // 2, 3, 3),
        )
        self.projection = nn.Sequential(nn.Dropout(0.0), nn.Linear(BACKBONE_DIM, EMBED_DIM))
        self.projection_dict = DictionaryProjection()
        self.eval()

    @torch.no_grad()
    def encode_clips(self, clips: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
        """(N, 3, 32, 224, 224) clips -> (N, 768) backbone features."""
        return torch.cat(
            [
                self.backbone(clips[i : i + chunk_size], mask_ratio=0.0)[0]
                for i in range(0, clips.shape[0], chunk_size)
            ]
        )

    @torch.no_grad()
    def encode_continuous(self, clips: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
        return F.normalize(self.projection(self.encode_clips(clips, chunk_size)), dim=-1)

    @torch.no_grad()
    def encode_dictionary(
        self, clips: torch.Tensor, lengths: torch.Tensor, chunk_size: int = 16
    ) -> torch.Tensor:
        return self.projection_dict(self.encode_clips(clips, chunk_size), lengths)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device="cpu") -> "SignMatch":
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(temporal_kernel=state[PATCH_KEY].shape[2])
        model.load_state_dict(state)
        return model.to(device).eval()


def match(queries: torch.Tensor, entries: torch.Tensor) -> torch.Tensor:
    return queries @ entries.T
