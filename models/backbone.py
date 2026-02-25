import torch
import torch.nn as nn
from torch import Tensor
import math

from .fusion import CodecFusionBlock

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class DiTConditionalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.fusion = CodecFusionBlock(d_model=d_model, n_heads=n_heads)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )

    def forward(self, x: Tensor, t_emb: Tensor, cond_xcodec: Tensor, cond_mae: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).unsqueeze(1).chunk(6, dim=-1)
        
        x_mod = x * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa * attn_out
        
        x_mod_fusion = x * (1 + scale_mlp) + shift_mlp
        fusion_out = self.fusion(x_mod_fusion, cond_xcodec, cond_mae)
        return x + gate_mlp * fusion_out

class LCFMBackbone(nn.Module):
    def __init__(self, in_channels: int, d_model: int, n_heads: int, depth: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.proj_in = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList([
            DiTConditionalBlock(d_model, n_heads) for _ in range(depth)
        ])
        
        self.norm_out = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, in_channels)

    def forward(self, x_t: Tensor, t: Tensor, cond_xcodec: Tensor, cond_mae: Tensor) -> Tensor:
        t = t.view(-1)
        t_emb = self.time_mlp(t)
        
        x = self.proj_in(x_t)
        
        for block in self.blocks:
            x = block(x, t_emb, cond_xcodec, cond_mae)
            
        x = self.norm_out(x)
        return self.proj_out(x)