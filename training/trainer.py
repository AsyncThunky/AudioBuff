import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from .ema import EMAShadowModel
from .checkpoint import CheckpointManager

class DDPTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        dataset: torch.utils.data.Dataset, 
        optimizer: torch.optim.Optimizer, 
        local_rank: int, 
        batch_size: int,
        checkpoint_dir: str | Path,
        resume_path: str | Path | None = None,
        ema_decay: float = 0.9999,
        num_workers: int = 4,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        log_interval: int = 25,
        checkpoint_meta: dict[str, int | float | str] | None = None,
    ):
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_amp = bool(use_amp and torch.cuda.is_available())
        self.log_interval = max(1, log_interval)
        self.checkpoint_meta = checkpoint_meta

        self.model = DDP(model.to(self.device), device_ids=[self.local_rank])
        self.optimizer = optimizer
        self.ema = EMAShadowModel(self.model.module, decay=ema_decay)
        self.checkpointer = CheckpointManager(checkpoint_dir)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp
        )
        
        self.start_epoch = 0
        if resume_path:
            self.start_epoch = self.checkpointer.load(
                resume_path, 
                self.model.module, 
                self.ema.get_model(), 
                self.optimizer
            ) + 1
            
        self.sampler = DistributedSampler(dataset)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=self.sampler, 
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.sampler.set_epoch(epoch)
        total_loss = 0.0

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.dataloader, start=1):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                loss = self.model(
                    dac_clean=batch['dac_clean'].to(self.device, non_blocking=True),
                    xcodec_noisy=batch['xcodec_noisy'].to(self.device, non_blocking=True),
                    mae_noisy=batch['mae_noisy'].to(self.device, non_blocking=True),
                )
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            should_step = (step % self.grad_accum_steps == 0) or (step == len(self.dataloader))
            if should_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model.module)

            total_loss += loss.item() * self.grad_accum_steps

            if self.local_rank == 0 and (step % self.log_interval == 0 or step == len(self.dataloader)):
                print(
                    f"Epoch {epoch} | Step {step}/{len(self.dataloader)} | "
                    f"Loss {loss.item() * self.grad_accum_steps:.6f}"
                )

        avg_loss = torch.tensor(total_loss / max(1, len(self.dataloader)), device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss / dist.get_world_size()
        avg_loss_value = float(avg_loss.item())
        
        if self.local_rank == 0:
            self.checkpointer.save(
                epoch, 
                self.model.module, 
                self.ema.get_model(), 
                self.optimizer,
                meta=self.checkpoint_meta,
            )
            
        return avg_loss_value
