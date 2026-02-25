import torch
import torch.nn as nn
from pathlib import Path


def _module_state_dict(module: nn.Module | None) -> dict[str, torch.Tensor] | None:
    if module is None:
        return None
    return module.state_dict()


class CheckpointManager:
    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self, 
        epoch: int, 
        model: nn.Module, 
        ema: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        is_best: bool = False,
        meta: dict[str, int | float | str] | None = None,
    ) -> None:
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'ema_state': ema.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'model_backbone_state': _module_state_dict(getattr(model, "vfn", None)),
            'model_aligner_state': _module_state_dict(getattr(model, "aligner", None)),
            'ema_backbone_state': _module_state_dict(getattr(ema, "vfn", None)),
            'ema_aligner_state': _module_state_dict(getattr(ema, "aligner", None)),
            'meta': meta or {},
        }
        
        torch.save(state, self.save_dir / f"checkpoint_epoch_{epoch}.pt")
        if is_best:
            torch.save(state, self.save_dir / "checkpoint_best.pt")

    def load(
        self, 
        checkpoint_path: str | Path, 
        model: nn.Module, 
        ema: nn.Module, 
        optimizer: torch.optim.Optimizer
    ) -> int:
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        if "model_state" in state:
            model.load_state_dict(state['model_state'])
        else:
            if "model_backbone_state" in state and hasattr(model, "vfn"):
                model.vfn.load_state_dict(state["model_backbone_state"])
            if "model_aligner_state" in state and hasattr(model, "aligner"):
                model.aligner.load_state_dict(state["model_aligner_state"])

        if "ema_state" in state:
            ema.load_state_dict(state['ema_state'])
        else:
            if "ema_backbone_state" in state and hasattr(ema, "vfn"):
                ema.vfn.load_state_dict(state["ema_backbone_state"])
            if "ema_aligner_state" in state and hasattr(ema, "aligner"):
                ema.aligner.load_state_dict(state["ema_aligner_state"])

        if "optimizer_state" in state:
            optimizer.load_state_dict(state['optimizer_state'])

        return int(state.get('epoch', 0))
