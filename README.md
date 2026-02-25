# AudioBuff: Multi-Codec Flow Matching Restoration

An advanced generative audio restoration pipeline. This architecture utilizes Latent Conditional Flow Matching (LCFM) via a Diffusion Transformer (DiT) backbone to reconstruct damaged audio signals.

By decoupling acoustic feature extraction from the generative regression loop, this model efficiently fuses semantic and acoustic context maps to repair severe audio degradation without hallucinating incorrect pitch or phonetic structures.

## Architecture

- **Target Codec:** Descript Audio Codec (DAC) - default `44khz` model profile.
- **Structural Conditioning:** X-Codec - low frame-rate macro-rhythm and pitch tracking.
- **Semantic Conditioning:** AudioMAE - contextual separation of signal versus noise.
- **Generative Backbone:** LCFM DiT with Adaptive Layer Normalization (AdaLN) and cross-attention multi-codec fusion.

## Repository Layout

```text
AudioBuff/
|-- data/
|   |-- dataset.py
|   `-- prepare_latents.py
|-- models/
|   |-- alignment.py
|   |-- backbone.py
|   `-- fusion.py
|-- training/
|   |-- checkpoint.py
|   |-- ema.py
|   |-- loss.py
|   `-- trainer.py
|-- inference/
|   |-- generate.py
|   `-- solver.py
|-- notebooks/
|   `-- AudioRepairAI_Colab.ipynb
|-- train.py
|-- requirements.txt
`-- README.md
```

## Installation

Use a modern Python environment (Python 3.10+ recommended), then install:

```bash
pip install -r requirements.txt
```

## Execution Pipeline

Training is split into two phases to reduce active VRAM pressure and make the workflow practical on constrained GPUs.

### Phase 1: Latent Extraction (Offline)

Pre-compute embeddings and cache tensors from degraded/clean audio pairs:

```bash
python -m data.prepare_latents
```

Common options:

```bash
python -m data.prepare_latents \
  --source_dir ./raw_pristine_audio \
  --out_dir ./latents \
  --dac_model_type 44khz \
  --segment_seconds 5.0 \
  --max_files 100
```

Expected default folders:
- Input pristine audio: `./raw_pristine_audio/*.wav`
- Output cached latents: `./latents/*.pt`

### Phase 2: Distributed Training

Train the vector field network on cached latents with DDP.

Single GPU (for example, Google Colab T4):

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py --profile smoke
```

Multi-GPU, single node:

```bash
torchrun --nproc_per_node=4 train.py
```

Available profiles:
- `smoke`: fastest sanity pass
- `poc`: balanced Colab default
- `full`: longer resumable training

Override any profile value with explicit CLI flags:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
  --profile poc \
  --epochs 20 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --amp
```

### Phase 3: Inference

Run ODE-based restoration on new damaged audio:

```bash
python -m inference.generate \
  --input damaged_track.wav \
  --output repaired_track.wav \
  --checkpoint checkpoints/checkpoint_best.pt \
  --cfg 2.0 \
  --steps 32 \
  --chunk_tokens 256 \
  --hop_tokens 128
```

The inference loader supports both:
- Legacy checkpoints with combined `ema_state` (prefixed `vfn.` and `aligner.` keys).
- New component checkpoints with `ema_backbone_state` and `ema_aligner_state`.

## Google Colab Notebook

Use `notebooks/AudioRepairAI_Colab.ipynb` for an end-to-end Colab workflow:

1. Mount Google Drive for persistent `latents/`, `checkpoints/`, and output artifacts.
2. Clone the repo and install dependencies.
3. Optionally install X-Codec from a user-supplied git URL.
4. Run `smoke`, `poc`, or `full` profile from one config cell.
5. Execute latent extraction, training, and inference in sequence.

## Performance and Compute Constraints

If you encounter OOM errors on 16GB GPUs:

1. Reduce `--batch_size`.
2. Increase `--grad_accum_steps`.
3. Use or keep `--amp` enabled.
4. Reduce model width/depth (`--d_model`, `--depth`).
