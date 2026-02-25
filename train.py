import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from data.dataset import CachedLatentDataset
from models.alignment import MultiCodecAlignment
from models.backbone import LCFMBackbone
from training.loss import CFGFlowMatchingLoss
from training.trainer import DDPTrainer


PROFILE_DEFAULTS: dict[str, dict[str, int | float | bool]] = {
    "smoke": {
        "epochs": 1,
        "batch_size": 2,
        "num_workers": 2,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "grad_accum_steps": 1,
        "amp": False,
    },
    "poc": {
        "epochs": 8,
        "batch_size": 4,
        "num_workers": 2,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "grad_accum_steps": 4,
        "amp": True,
    },
    "full": {
        "epochs": 100,
        "batch_size": 8,
        "num_workers": 4,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "grad_accum_steps": 8,
        "amp": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-codec LCFM DiT with DDP.")
    parser.add_argument("--profile", type=str, default="poc", choices=tuple(PROFILE_DEFAULTS))

    parser.add_argument("--data_dir", type=Path, default=Path("./latents"))
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--resume_path", type=Path, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1337)

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)

    parser.add_argument("--in_channels", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--xcodec_dim", type=int, default=512)
    parser.add_argument("--mae_dim", type=int, default=768)
    parser.add_argument("--p_uncond", type=float, default=0.15)
    return parser.parse_args()


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = PROFILE_DEFAULTS[args.profile]
    for key in ("epochs", "batch_size", "num_workers", "lr", "weight_decay", "grad_accum_steps"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    if args.amp is None:
        args.amp = bool(defaults["amp"])
    return args


def setup_distributed() -> int:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, local_rank: int) -> None:
    rank_seed = seed + local_rank
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)


def main() -> None:
    args = apply_profile_defaults(parse_args())
    local_rank = setup_distributed()
    seed_everything(args.seed, local_rank)
    device = torch.device(f"cuda:{local_rank}")

    try:
        vector_field_network = LCFMBackbone(
            in_channels=args.in_channels,
            d_model=args.d_model,
            n_heads=args.n_heads,
            depth=args.depth,
        )
        aligner = MultiCodecAlignment(
            xcodec_dim=args.xcodec_dim,
            mae_dim=args.mae_dim,
            dac_dim=args.in_channels,
        )
        loss_module = CFGFlowMatchingLoss(
            vector_field_network=vector_field_network,
            aligner=aligner,
            p_uncond=args.p_uncond,
        ).to(device)

        optimizer = torch.optim.AdamW(
            loss_module.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        dataset = CachedLatentDataset(data_dir=args.data_dir)

        checkpoint_meta = {
            "profile": args.profile,
            "in_channels": args.in_channels,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "depth": args.depth,
            "xcodec_dim": args.xcodec_dim,
            "mae_dim": args.mae_dim,
            "p_uncond": args.p_uncond,
        }

        trainer = DDPTrainer(
            model=loss_module,
            dataset=dataset,
            optimizer=optimizer,
            local_rank=local_rank,
            batch_size=int(args.batch_size),
            checkpoint_dir=args.checkpoint_dir,
            resume_path=args.resume_path,
            ema_decay=args.ema_decay,
            num_workers=int(args.num_workers),
            grad_accum_steps=int(args.grad_accum_steps),
            use_amp=bool(args.amp),
            log_interval=args.log_interval,
            checkpoint_meta=checkpoint_meta,
        )

        for epoch in range(trainer.start_epoch, int(args.epochs)):
            avg_loss = trainer.train_epoch(epoch)
            if local_rank == 0:
                print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
