import argparse
import random
from pathlib import Path
from typing import Any

import torch
import torchaudio
import torchaudio.functional as F

# Note: Import paths depend on your installed codec libraries
import dac
from transformers import AutoModel, AutoFeatureExtractor


MODEL_SAMPLE_RATES: dict[str, int] = {
    "16khz": 16000,
    "24khz": 24000,
    "44khz": 44100,
}


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
    raise TypeError("No tensor could be extracted from codec output.")


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


def _resolve_sample_rate(dac_model_type: str, sample_rate: int | None) -> int:
    if sample_rate is not None:
        return sample_rate
    if dac_model_type not in MODEL_SAMPLE_RATES:
        raise ValueError(
            f"Unknown dac_model_type '{dac_model_type}'. "
            f"Supported: {sorted(MODEL_SAMPLE_RATES)}"
        )
    return MODEL_SAMPLE_RATES[dac_model_type]


def _iter_segments(waveform: torch.Tensor, segment_samples: int) -> list[torch.Tensor]:
    if waveform.ndim != 2 or waveform.size(0) != 1:
        raise ValueError(f"Expected mono waveform [1, samples], got {tuple(waveform.shape)}")

    segments: list[torch.Tensor] = []
    total_samples = waveform.size(-1)
    if total_samples == 0:
        return segments

    for start in range(0, total_samples, segment_samples):
        end = min(start + segment_samples, total_samples)
        segment = waveform[:, start:end]
        if segment.size(-1) < segment_samples:
            pad = segment_samples - segment.size(-1)
            segment = torch.nn.functional.pad(segment, (0, pad))
        segments.append(segment)
    return segments


class DegradationEngine:
    def __init__(self, sample_rate: int):
        self.sr = sample_rate

    def apply_clipping(self, waveform: torch.Tensor, overdrive: float = 2.0) -> torch.Tensor:
        clipped = waveform * overdrive
        return torch.clamp(clipped, min=-1.0, max=1.0)

    def apply_bad_mic_eq(self, waveform: torch.Tensor) -> torch.Tensor:
        # Simulates a cheap microphone by cutting extreme lows and highs.
        waveform = F.highpass_biquad(waveform, self.sr, cutoff_freq=200.0)
        waveform = F.lowpass_biquad(waveform, self.sr, cutoff_freq=8000.0)
        return waveform

    def add_noise(self, waveform: torch.Tensor, snr_db: float = 15.0) -> torch.Tensor:
        noise = torch.randn_like(waveform)
        signal_power = waveform.norm(p=2)
        noise_power = noise.norm(p=2)

        snr = 10 ** (snr_db / 20)
        scale = signal_power / (snr * noise_power + 1e-8)
        return waveform + noise * scale

    def ruin_audio(self, clean_waveform: torch.Tensor) -> torch.Tensor:
        noisy = self.apply_bad_mic_eq(clean_waveform)
        noisy = self.apply_clipping(noisy, overdrive=torch.empty(1).uniform_(1.5, 3.0).item())
        noisy = self.add_noise(noisy, snr_db=torch.empty(1).uniform_(10.0, 25.0).item())
        return noisy


class LatentExtractor:
    def __init__(self, device: torch.device, sample_rate: int, dac_model_type: str):
        self.device = device
        self.sample_rate = sample_rate
        self.xcodec_feature_dim = 512

        self.dac_model = dac.DAC.load(dac.utils.download(model_type=dac_model_type)).to(device)
        self.dac_model.eval()
        self.dac_latent_dim = int(getattr(self.dac_model, "latent_dim", 1024))

        self.mae_extractor = AutoFeatureExtractor.from_pretrained("facebook/audiomae-base")
        self.mae_model = AutoModel.from_pretrained("facebook/audiomae-base").to(device)
        self.mae_model.eval()
        self.mae_hidden_dim = int(getattr(self.mae_model.config, "hidden_size", 768))

        self.xcodec_model = self._try_load_xcodec()

    def _try_load_xcodec(self) -> Any | None:
        try:
            import xcodec  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on external install
            print(f"[prepare_latents] X-Codec unavailable ({exc}). Using random fallback latents.")
            return None

        if hasattr(xcodec, "load_model"):
            try:
                model = xcodec.load_model()
                if hasattr(model, "to"):
                    model = model.to(self.device)
                if hasattr(model, "eval"):
                    model.eval()
                print("[prepare_latents] Loaded X-Codec using xcodec.load_model().")
                return model
            except Exception as exc:  # pragma: no cover - depends on external install
                print(
                    f"[prepare_latents] Failed to initialize X-Codec from load_model() ({exc}). "
                    "Using random fallback latents."
                )
                return None

        print(
            "[prepare_latents] X-Codec module found but no supported load_model() API. "
            "Using random fallback latents."
        )
        return None

    def _encode_xcodec(self, noisy_waveform: torch.Tensor) -> torch.Tensor:
        seq_len = max(1, noisy_waveform.size(-1) // 320)
        if self.xcodec_model is None:
            return torch.randn((seq_len, self.xcodec_feature_dim), device=self.device)

        if hasattr(self.xcodec_model, "encode"):
            try:
                xcodec_emb = self.xcodec_model.encode(noisy_waveform.unsqueeze(0))
                return _to_sequence_features(
                    xcodec_emb,
                    name="xcodec_noisy",
                    expected_feature_dim=self.xcodec_feature_dim,
                )
            except Exception as exc:  # pragma: no cover - depends on external install
                print(
                    f"[prepare_latents] X-Codec encode failed ({exc}). "
                    "Using random fallback latents for this segment."
                )
        return torch.randn((seq_len, self.xcodec_feature_dim), device=self.device)

    @torch.inference_mode()
    def process_segment(self, clean_waveform: torch.Tensor, engine: DegradationEngine) -> dict[str, torch.Tensor]:
        clean_waveform = clean_waveform.to(self.device)
        noisy_waveform = engine.ruin_audio(clean_waveform)

        dac_encoded = self.dac_model.encode(clean_waveform.unsqueeze(0))
        dac_clean = _to_sequence_features(
            dac_encoded,
            name="dac_clean",
            expected_feature_dim=self.dac_latent_dim,
        )

        xcodec_noisy = self._encode_xcodec(noisy_waveform)

        mae_inputs = self.mae_extractor(
            noisy_waveform.squeeze(0).cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).to(self.device)
        mae_output = self.mae_model(**mae_inputs).last_hidden_state
        mae_noisy = _to_sequence_features(
            mae_output,
            name="mae_noisy",
            expected_feature_dim=self.mae_hidden_dim,
        )

        return {
            "dac_clean": dac_clean.to(dtype=torch.float32, device="cpu"),
            "xcodec_noisy": xcodec_noisy.to(dtype=torch.float32, device="cpu"),
            "mae_noisy": mae_noisy.to(dtype=torch.float32, device="cpu"),
        }


def build_dataset(
    source_dir: Path,
    out_dir: Path,
    *,
    sample_rate: int,
    segment_seconds: float,
    max_files: int | None,
    dac_model_type: str,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segment_samples = max(1, int(sample_rate * segment_seconds))

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    extractor = LatentExtractor(device=device, sample_rate=sample_rate, dac_model_type=dac_model_type)
    engine = DegradationEngine(sample_rate=sample_rate)

    wav_files = sorted(source_dir.glob("*.wav"))
    if max_files is not None:
        wav_files = wav_files[:max_files]

    if not wav_files:
        print(f"[prepare_latents] No .wav files found in {source_dir}")
        return

    total_saved = 0
    total_failed = 0

    for file_idx, wav_path in enumerate(wav_files, start=1):
        try:
            waveform, sr = torchaudio.load(wav_path)
            if sr != sample_rate:
                waveform = F.resample(waveform, sr, sample_rate)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            segments = _iter_segments(waveform, segment_samples=segment_samples)
            if not segments:
                print(f"[prepare_latents] Skipping empty file: {wav_path.name}")
                continue

            for segment_idx, segment in enumerate(segments):
                latents = extractor.process_segment(segment, engine)
                save_name = f"{wav_path.stem}_seg_{segment_idx:05d}.pt"
                torch.save(latents, out_dir / save_name)
                total_saved += 1

            print(
                f"[prepare_latents] Processed {file_idx}/{len(wav_files)} files | "
                f"{wav_path.name} -> {len(segments)} segments"
            )
        except Exception as exc:
            total_failed += 1
            print(f"[prepare_latents] Failed {wav_path.name}: {exc}")

    print(
        f"[prepare_latents] Completed. Saved segments: {total_saved}, "
        f"failed files: {total_failed}, output: {out_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute cached latents from clean audio.")
    parser.add_argument("--source_dir", type=Path, default=Path("./raw_pristine_audio"))
    parser.add_argument("--out_dir", type=Path, default=Path("./latents"))
    parser.add_argument("--sample_rate", type=int, default=None)
    parser.add_argument("--segment_seconds", type=float, default=5.0)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--dac_model_type", type=str, default="44khz")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_rate = _resolve_sample_rate(args.dac_model_type, args.sample_rate)
    build_dataset(
        source_dir=args.source_dir,
        out_dir=args.out_dir,
        sample_rate=sample_rate,
        segment_seconds=args.segment_seconds,
        max_files=args.max_files,
        dac_model_type=args.dac_model_type,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
