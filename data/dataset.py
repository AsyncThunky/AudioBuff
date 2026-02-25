import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path

class CachedLatentDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        self.files = sorted(Path(data_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No latent .pt files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = torch.load(self.files[idx], weights_only=True)
        if not isinstance(sample, dict):
            raise TypeError(f"Malformed latent sample: {self.files[idx]} does not contain a dict.")

        required_keys = ("dac_clean", "xcodec_noisy", "mae_noisy")
        missing = [key for key in required_keys if key not in sample]
        if missing:
            raise KeyError(f"Malformed latent sample: {self.files[idx]} missing keys {missing}")

        output: dict[str, Tensor] = {}
        for key in required_keys:
            value = sample[key]
            if not isinstance(value, Tensor):
                raise TypeError(f"Malformed latent sample: {self.files[idx]} key '{key}' is not a tensor")
            if value.ndim != 2:
                raise ValueError(
                    f"Malformed latent sample: {self.files[idx]} key '{key}' expected 2D tensor, "
                    f"got shape {tuple(value.shape)}"
                )
            output[key] = value.to(dtype=torch.float32)
        return output
