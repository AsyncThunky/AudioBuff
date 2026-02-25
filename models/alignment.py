import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SequenceAligner(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor, target_len: int) -> Tensor:
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x.transpose(1, 2)

class MultiCodecAlignment(nn.Module):
    def __init__(self, xcodec_dim: int, mae_dim: int, dac_dim: int):
        super().__init__()
        self.xcodec_aligner = SequenceAligner(xcodec_dim, dac_dim)
        self.mae_aligner = SequenceAligner(mae_dim, dac_dim)

    def forward(self, target_latent: Tensor, xcodec_emb: Tensor, mae_emb: Tensor) -> tuple[Tensor, Tensor]:
        target_len = target_latent.size(1)
        xcodec_aligned = self.xcodec_aligner(xcodec_emb, target_len)
        mae_aligned = self.mae_aligner(mae_emb, target_len)
        return xcodec_aligned, mae_aligned