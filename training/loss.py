import torch
import torch.nn as nn
from torch import Tensor

class CFGFlowMatchingLoss(nn.Module):
    def __init__(self, vector_field_network: nn.Module, aligner: nn.Module, p_uncond: float = 0.15):
        super().__init__()
        self.vfn = vector_field_network
        self.aligner = aligner
        self.p_uncond = p_uncond
        self.mse = nn.MSELoss()

    def forward(self, dac_clean: Tensor, xcodec_noisy: Tensor, mae_noisy: Tensor) -> Tensor:
        batch_size = dac_clean.size(0)
        device = dac_clean.device
        
        xcodec_aligned, mae_aligned = self.aligner(dac_clean, xcodec_noisy, mae_noisy)
        
        if self.training and self.p_uncond > 0.0:
            mask = torch.rand(batch_size, 1, 1, device=device) > self.p_uncond
            mask = mask.to(xcodec_aligned.dtype)
            xcodec_aligned = xcodec_aligned * mask
            mae_aligned = mae_aligned * mask
            
        x_0 = torch.randn_like(dac_clean)
        x_1 = dac_clean
        t = torch.rand((batch_size, 1, 1), device=device)
        
        x_t = t * x_1 + (1 - t) * x_0
        target_velocity = x_1 - x_0
        
        predicted_velocity = self.vfn(
            x_t=x_t, 
            t=t, 
            cond_xcodec=xcodec_aligned, 
            cond_mae=mae_aligned
        )
        
        return self.mse(predicted_velocity, target_velocity)