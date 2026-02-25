import argparse
from pathlib import Path
from typing import Any

import torch
import torchaudio

# Note: Import paths depend on your installed codec libraries
import dac
from transformers import AutoModel, AutoFeatureExtractor

from models.backbone import LCFMBackbone
from models.alignment import MultiCodecAlignment
from inference.solver import CFGFlowODESolver, StreamingAudioProcessor


MODEL_SAMPLE_RATES: dict[str, int] = {
    "16khz": 16000,
    "24khz": 24000,
    "44khz": 44100,
}


def _resolve_sample_rate(dac_model_type: str, sample_rate: int | None) -> int:
    if sample_rate is not None:
        return sample_rate
    if dac_model_type not in MODEL_SAMPLE_RATES:
        raise ValueError(
            f"Unknown dac_model_type '{dac_model_type}'. Supported: {sorted(MODEL_SAMPLE_RATES)}"
        )
    return MODEL_SAMPLE_RATES[dac_model_type]


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
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


def _to_sequence_features(
    value: Any,
    *,
    name: str,
    expected_feature_dim: int | None = None,
) -> torch.Tensor:
    tensor = _first_tensor(value)
    if tensor.ndim == 3:
        if tensor.size(0) != 1:
            raise ValueError(f"{name} expected batch size 1, got shape {tuple(tensor.shape)}")
        tensor = tensor.squeeze(0)
    elif tensor.ndim != 2:
        raise ValueError(f"{name} expected 2D or 3D tensor, got shape {tuple(tensor.shape)}")

    if expected_feature_dim is not None:
        if tensor.size(-1) == expected_feature_dim:
            return tensor.contiguous()
        if tensor.size(0) == expected_feature_dim:
            return tensor.transpose(0, 1).contiguous()

    if tensor.size(0) > tensor.size(1):
        tensor = tensor.transpose(0, 1)
    return tensor.contiguous()


def _strip_prefixed_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


class AudioRestorationPipeline:
    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        *,
        dac_model_type: str = "44khz",
        sample_rate: int | None = None,
        chunk_tokens: int = 256,
        hop_tokens: int = 128,
        steps: int = 32,
        in_channels: int = 1024,
        d_model: int = 1024,
        n_heads: int = 16,
        depth: int = 12,
        xcodec_dim: int = 512,
        mae_dim: int = 768,
    ):
        self.device = device
        self.steps = steps
        self.sr = _resolve_sample_rate(dac_model_type, sample_rate)
        self.xcodec_dim = xcodec_dim
        self.mae_dim = mae_dim

        # 1. Initialize Feature Extractors.
        self.dac_model = dac.DAC.load(dac.utils.download(model_type=dac_model_type)).to(device)
        self.dac_model.eval()

        self.xcodec_model = self._try_load_xcodec()
        self.mae_extractor = AutoFeatureExtractor.from_pretrained("facebook/audiomae-base")
        self.mae_model = AutoModel.from_pretrained("facebook/audiomae-base").to(device)
        self.mae_model.eval()
        self.mae_hidden_dim = int(getattr(self.mae_model.config, "hidden_size", mae_dim))

        # 2. Initialize Generative Backbone and aligner.
        self.backbone = LCFMBackbone(
            in_channels=in_channels,
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
        ).to(device)
        self.aligner = MultiCodecAlignment(
            xcodec_dim=xcodec_dim,
            mae_dim=mae_dim,
            dac_dim=in_channels,
        ).to(device)

        self._load_checkpoint_weights(checkpoint_path)
        self.backbone.eval()
        self.aligner.eval()

        # 3. Initialize ODE solver and chunk processor.
        self.solver = CFGFlowODESolver(self.backbone, self.aligner, self.dac_model)
        self.processor = StreamingAudioProcessor(
            solver=self.solver,
            chunk_length=max(1, chunk_tokens),
            hop_length=max(1, hop_tokens),
        )

    def _try_load_xcodec(self) -> Any | None:
        try:
            import xcodec  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional install
            print(f"[inference] X-Codec unavailable ({exc}). Falling back to random conditioning.")
            return None

        if hasattr(xcodec, "load_model"):
            try:
                model = xcodec.load_model()
                if hasattr(model, "to"):
                    model = model.to(self.device)
                if hasattr(model, "eval"):
                    model.eval()
                print("[inference] Loaded X-Codec using xcodec.load_model().")
                return model
            except Exception as exc:  # pragma: no cover
                print(f"[inference] X-Codec load_model() failed ({exc}). Falling back to random conditioning.")
                return None

        print("[inference] X-Codec module found but no supported load_model() API. Falling back to random conditioning.")
        return None

    def _load_checkpoint_weights(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)

        if "ema_backbone_state" in state and state["ema_backbone_state"] is not None:
            self.backbone.load_state_dict(state["ema_backbone_state"], strict=True)
            if "ema_aligner_state" in state and state["ema_aligner_state"] is not None:
                self.aligner.load_state_dict(state["ema_aligner_state"], strict=True)
            elif "model_aligner_state" in state and state["model_aligner_state"] is not None:
                self.aligner.load_state_dict(state["model_aligner_state"], strict=True)
            print("[inference] Loaded component EMA checkpoint schema.")
            return

        if "ema_state" in state:
            ema_state = state["ema_state"]
            backbone_state = _strip_prefixed_state(ema_state, "vfn.")
            aligner_state = _strip_prefixed_state(ema_state, "aligner.")

            if backbone_state:
                self.backbone.load_state_dict(backbone_state, strict=True)
            else:
                self.backbone.load_state_dict(ema_state, strict=False)
            if aligner_state:
                self.aligner.load_state_dict(aligner_state, strict=True)
            elif "model_state" in state:
                model_aligner_state = _strip_prefixed_state(state["model_state"], "aligner.")
                if model_aligner_state:
                    self.aligner.load_state_dict(model_aligner_state, strict=True)
            print("[inference] Loaded legacy combined ema_state checkpoint schema.")
            return

        if "model_state" in state:
            model_state = state["model_state"]
            backbone_state = _strip_prefixed_state(model_state, "vfn.")
            aligner_state = _strip_prefixed_state(model_state, "aligner.")
            if backbone_state:
                self.backbone.load_state_dict(backbone_state, strict=True)
            if aligner_state:
                self.aligner.load_state_dict(aligner_state, strict=True)
            print("[inference] Loaded model_state checkpoint schema (no EMA present).")
            return

        raise KeyError("Checkpoint missing compatible keys (ema_state or component EMA states).")

    def _encode_xcodec(self, wav: torch.Tensor) -> torch.Tensor:
        seq_len = max(1, wav.size(-1) // 320)
        if self.xcodec_model is None:
            return torch.randn((1, seq_len, self.xcodec_dim), device=self.device)

        if hasattr(self.xcodec_model, "encode"):
            try:
                xcodec_emb = self.xcodec_model.encode(wav.unsqueeze(0))
                xcodec_seq = _to_sequence_features(
                    xcodec_emb,
                    name="xcodec_noisy",
                    expected_feature_dim=self.xcodec_dim,
                )
                return xcodec_seq.unsqueeze(0)
            except Exception as exc:  # pragma: no cover - depends on optional install
                print(f"[inference] X-Codec encode failed ({exc}). Falling back to random conditioning.")
        return torch.randn((1, seq_len, self.xcodec_dim), device=self.device)

    @torch.inference_mode()
    def repair_file(self, input_path: Path, output_path: Path, cfg_scale: float = 1.5) -> None:
        wav, sr = torchaudio.load(input_path)

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.to(self.device)

        xcodec_noisy = self._encode_xcodec(wav)

        mae_inputs = self.mae_extractor(
            wav.squeeze(0).cpu().numpy(),
            sampling_rate=self.sr,
            return_tensors="pt"
        ).to(self.device)
        mae_hidden = self.mae_model(**mae_inputs).last_hidden_state
        mae_noisy = _to_sequence_features(
            mae_hidden,
            name="mae_noisy",
            expected_feature_dim=self.mae_hidden_dim,
        ).unsqueeze(0)

        repaired_waveform = self.processor.process_full_track(
            full_xcodec=xcodec_noisy,
            full_mae=mae_noisy,
            cfg_scale=cfg_scale,
            steps=self.steps,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, repaired_waveform.cpu().squeeze(0), self.sr)
        print(f"Repaired audio saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Audio Stem Restoration")
    parser.add_argument("--input", type=str, required=True, help="Path to damaged .wav file")
    parser.add_argument("--output", type=str, required=True, help="Path to save repaired .wav file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--cfg", type=float, default=1.5, help="CFG Scale (higher = more aggressive repair)")
    parser.add_argument("--steps", type=int, default=32, help="ODE solver integration steps")
    parser.add_argument("--chunk_tokens", type=int, default=256, help="Token chunk size for streaming")
    parser.add_argument("--hop_tokens", type=int, default=128, help="Token hop size for streaming")
    parser.add_argument("--dac_model_type", type=str, default="44khz", help="DAC model type (16khz|24khz|44khz)")
    parser.add_argument("--sample_rate", type=int, default=None, help="Optional explicit sample rate override")
    parser.add_argument("--in_channels", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--xcodec_dim", type=int, default=512)
    parser.add_argument("--mae_dim", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = AudioRestorationPipeline(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        dac_model_type=args.dac_model_type,
        sample_rate=args.sample_rate,
        chunk_tokens=args.chunk_tokens,
        hop_tokens=args.hop_tokens,
        steps=args.steps,
        in_channels=args.in_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        xcodec_dim=args.xcodec_dim,
        mae_dim=args.mae_dim,
    )
    pipeline.repair_file(Path(args.input), Path(args.output), args.cfg)


if __name__ == "__main__":
    main()
