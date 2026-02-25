import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _first_tensor(value: object) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError("No tensor output found.")


class CFGFlowODESolver(nn.Module):
    def __init__(self, vector_field_network: nn.Module, aligner: nn.Module, dac_decoder: nn.Module):
        super().__init__()
        self.vfn = vector_field_network
        self.aligner = aligner
        self.dac = dac_decoder

    def _latent_dim(self) -> int:
        if hasattr(self.dac, "latent_dim"):
            return int(self.dac.latent_dim)
        if hasattr(self.vfn, "proj_out") and hasattr(self.vfn.proj_out, "out_features"):
            return int(self.vfn.proj_out.out_features)
        raise ValueError("Unable to infer latent_dim for ODE generation.")

    @torch.inference_mode()
    def generate_latents(
        self, xcodec_noisy: Tensor, mae_noisy: Tensor, target_seq_len: int, 
        cfg_scale: float = 1.5, steps: int = 32
    ) -> Tensor:
        
        batch_size = xcodec_noisy.size(0)
        device = xcodec_noisy.device
        
        dummy_target = torch.empty((batch_size, target_seq_len, self._latent_dim()), device=device)
        cond_xcodec, cond_mae = self.aligner(dummy_target, xcodec_noisy, mae_noisy)
        
        uncond_xcodec = torch.zeros_like(cond_xcodec)
        uncond_mae = torch.zeros_like(cond_mae)
        
        x_t = torch.randn_like(dummy_target)
        dt = 1.0 / steps
        
        for step in range(steps):
            t_val = step * dt
            t_tensor = torch.full((batch_size, 1, 1), t_val, device=device)
            
            v_cond = self.vfn(x_t, t_tensor, cond_xcodec, cond_mae)
            
            if cfg_scale != 1.0:
                v_uncond = self.vfn(x_t, t_tensor, uncond_xcodec, uncond_mae)
                velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                velocity = v_cond
                
            x_t = x_t + velocity * dt
            
        return x_t

class StreamingAudioProcessor:
    def __init__(self, solver: CFGFlowODESolver, chunk_length: int, hop_length: int):
        self.solver = solver
        self.chunk_length = chunk_length
        self.hop_length = hop_length

    def _dac_hop_size(self) -> int:
        if hasattr(self.solver.dac, "hop_size"):
            return int(self.solver.dac.hop_size)
        return 512

    def _decode_latents(self, repaired_latents: Tensor) -> Tensor:
        # DAC decode typically expects [B, D, T] latents.
        decoded = self.solver.dac.decode(repaired_latents.transpose(1, 2).contiguous())
        waveform = _first_tensor(decoded)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3:
            raise ValueError(f"Unexpected decode output shape: {tuple(waveform.shape)}")
        return waveform

    @torch.inference_mode()
    def process_full_track(
        self,
        full_xcodec: Tensor,
        full_mae: Tensor,
        cfg_scale: float,
        steps: int = 32,
    ) -> Tensor:
        if full_xcodec.ndim != 3 or full_mae.ndim != 3:
            raise ValueError(
                "Expected full_xcodec/full_mae shaped [batch, tokens, channels], got "
                f"{tuple(full_xcodec.shape)} and {tuple(full_mae.shape)}"
            )

        device = full_xcodec.device
        batch_size = full_xcodec.size(0)
        total_tokens = min(full_xcodec.size(1), full_mae.size(1))
        if total_tokens <= 0:
            raise ValueError("Input conditioning sequences are empty.")

        chunk_tokens = max(1, min(self.chunk_length, total_tokens))
        hop_tokens = max(1, min(self.hop_length, chunk_tokens))
        dac_hop_size = self._dac_hop_size()

        total_samples = max(1, total_tokens * dac_hop_size)
        out_waveform = torch.zeros((batch_size, 1, total_samples), device=device)
        window_sum = torch.zeros_like(out_waveform)

        for start_idx in range(0, total_tokens, hop_tokens):
            end_idx = min(start_idx + chunk_tokens, total_tokens)
            actual_tokens = end_idx - start_idx

            chunk_xcodec = full_xcodec[:, start_idx:end_idx, :]
            chunk_mae = full_mae[:, start_idx:end_idx, :]

            if actual_tokens < chunk_tokens:
                pad_tokens = chunk_tokens - actual_tokens
                chunk_xcodec = F.pad(chunk_xcodec, (0, 0, 0, pad_tokens))
                chunk_mae = F.pad(chunk_mae, (0, 0, 0, pad_tokens))

            repaired_latents = self.solver.generate_latents(
                chunk_xcodec,
                chunk_mae,
                target_seq_len=chunk_tokens,
                cfg_scale=cfg_scale,
                steps=steps,
            )
            chunk_waveform = self._decode_latents(repaired_latents)

            keep_samples = max(1, actual_tokens * dac_hop_size)
            chunk_waveform = chunk_waveform[..., :keep_samples]
            if chunk_waveform.size(-1) == 0:
                continue

            if hop_tokens < chunk_tokens and chunk_waveform.size(-1) > 1:
                window = torch.hann_window(
                    chunk_waveform.size(-1), device=device, periodic=False
                ).view(1, 1, -1)
            else:
                window = torch.ones((1, 1, chunk_waveform.size(-1)), device=device)

            windowed_chunk = chunk_waveform * window
            start_sample = start_idx * dac_hop_size
            end_sample = min(start_sample + windowed_chunk.size(-1), total_samples)
            write_len = end_sample - start_sample
            if write_len <= 0:
                continue

            out_waveform[..., start_sample:end_sample] += windowed_chunk[..., :write_len]
            window_sum[..., start_sample:end_sample] += window[..., :write_len]

        return out_waveform / torch.clamp(window_sum, min=1e-8)
