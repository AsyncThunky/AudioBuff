import torch
import torch.nn as nn
from torch import Tensor

class CodecFusionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_xcodec = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_mae = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, dac_latent: Tensor, xcodec_emb: Tensor, mae_emb: Tensor) -> Tensor:
        attn_out_x, _ = self.cross_attn_xcodec(
            query=dac_latent, key=xcodec_emb, value=xcodec_emb
        )
        x = self.norm1(dac_latent + attn_out_x)

        attn_out_mae, _ = self.cross_attn_mae(
            query=x, key=mae_emb, value=mae_emb
        )
        x = self.norm2(x + attn_out_mae)

        ffn_out = self.ffn(x)
        return self.norm3(x + ffn_out)