import torch
import torch.nn as nn
from copy import deepcopy

class EMAShadowModel:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, active_model: nn.Module) -> None:
        shadow_params = dict(self.shadow.named_parameters())
        active_params = dict(active_model.named_parameters())

        for name, param in active_params.items():
            if param.requires_grad:
                shadow_params[name].copy_(
                    self.decay * shadow_params[name] + (1.0 - self.decay) * param
                )

        shadow_buffers = dict(self.shadow.named_buffers())
        active_buffers = dict(active_model.named_buffers())

        for name, buffer in active_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def get_model(self) -> nn.Module:
        return self.shadow