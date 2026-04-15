"""
Drop-in optimizer & scheduler setup for FGIC fine-tuning.

Features:
- AdamW param groups:
  * no weight decay for norm layers / bias
  * larger LR for classification head (e.g., 'head', 'classifier', 'fc')
  * (optional) layer-wise LR decay (LLRD) via regex rules
- Schedulers:
  * 'warmup_cosine'  (per-step warmup + cosine, 推荐)
  * 'flat_cosine'    (前期常数 LR，后期余弦)
  * 'warm_restarts'  (CosineAnnealingWarmRestarts)
  * 'plateau'        (ReduceLROnPlateau，用于验证指标停滞时降 LR)
- Stepper:
  * batch_step():  每个 batch 调用（自动 clip grad / optimizer.step / scheduler.step）
  * epoch_step():  每个 epoch 末尾调用（Plateau/epoch-based 调度）
"""

from typing import List, Tuple, Optional, Iterable, Dict
import re
import numpy as np
import torch
import torch.nn as nn


# ------------------------------- Param Groups -------------------------------

def _unwrap_module(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)

def build_param_groups(
    model: nn.Module,
    base_lr: float,
    wd: float = 0.05,
    head_lr_mult: float = 3.0,
    head_keywords: Tuple[str, ...] = ("head", "classifier", "fc"),
    norm_wd: float = 0.0,
    bias_wd: float = 0.0,
) -> List[dict]:
    """
    Return AdamW param groups with:
      - larger LR for classification head
      - no weight decay for norm & bias
    """
    m = _unwrap_module(model)
    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)
    head_params, decay_params, nodecay_params = [], [], []

    for module_name, module in m.named_modules():
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            full_name = f"{module_name}.{pname}" if module_name else pname
            is_head = any(k in module_name.lower() or k in full_name.lower()
                          for k in head_keywords)
            is_norm = isinstance(module, norm_types)
            is_bias = pname.endswith("bias")

            if is_head:
                head_params.append(p)
            elif is_norm:
                nodecay_params.append(p)
            elif is_bias:
                nodecay_params.append(p)
            else:
                decay_params.append(p)

    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": base_lr * head_lr_mult, "weight_decay": wd})
    if nodecay_params:
        groups.append({"params": nodecay_params, "lr": base_lr, "weight_decay": 0.0})
    if decay_params:
        groups.append({"params": decay_params, "lr": base_lr, "weight_decay": wd})

    # set initial_lr for each group (LambdaLR uses this)
    for g in groups:
        g.setdefault("initial_lr", g["lr"])
    return groups


def apply_llrd(
    model: nn.Module,
    param_groups: List[dict],
    rules: Optional[List[Tuple[str, float]]] = None,
) -> List[dict]:
    """
    Optional Layer-wise LR Decay (LLRD).

    rules = [
      (r"stages\.0\.", 0.9**3),
      (r"stages\.1\.", 0.9**2),
      (r"stages\.2\.", 0.9**1),
      (r"stages\.3\.", 1.0),
    ]

    You can customize patterns to your model (ConvNeXtV2 uses 'stages.i').
    """
    if not rules:
        return param_groups

    compiled = [(re.compile(p), mult) for p, mult in rules]
    m = _unwrap_module(model)

    # expand groups to per-param groups so each param can get its own lr mult
    expanded: List[dict] = []
    for g in param_groups:
        base_lr = g["lr"]
        for p in g["params"]:
            name = None
            # try to find param name (optional; if not found, keep base_lr)
            for n, pp in m.named_parameters():
                if pp is p:
                    name = n
                    break
            lr_mult = 1.0
            if name is not None:
                for pattern, mult in compiled:
                    if pattern.search(name):
                        lr_mult = mult
                        break
            expanded.append({
                "params": [p],
                "lr": base_lr * lr_mult,
                "weight_decay": g["weight_decay"],
                "initial_lr": base_lr * lr_mult,
            })
    return expanded


# ------------------------------- Schedulers ---------------------------------

def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    warmup_init_factor: float = 0.01,
) -> Tuple[torch.optim.lr_scheduler._LRScheduler, bool]:
    """
    Per-step linear warmup -> cosine decay to min_lr.
    Returns (scheduler, step_on_batch=True).
    """
    total_steps = max(1, total_epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def make_lambda(lr0: float):
        floor = min_lr / max(lr0, 1e-12)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # from warmup_init_factor -> 1.0
                return max(floor, warmup_init_factor + (1.0 - warmup_init_factor) * (step / warmup_steps))
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(floor, floor + (1.0 - floor) * cosine)
        return lr_lambda

    lambdas = [make_lambda(lr0) for lr0 in base_lrs]
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    return sched, True


def _build_flat_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    flat_ratio: float = 0.3,
    min_lr: float = 1e-6,
) -> Tuple[torch.optim.lr_scheduler._LRScheduler, bool]:
    """
    Epoch-level: keep LR constant for flat_ratio, then cosine.
    Returns (scheduler, step_on_batch=False).
    """
    flat_epochs = int(total_epochs * flat_ratio)
    cos_epochs = max(1, total_epochs - flat_epochs)
    flat = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=flat_epochs)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cos_epochs, eta_min=min_lr)
    sched = torch.optim.lr_scheduler.SequentialLR(optimizer, [flat, cos], milestones=[flat_epochs])
    return sched, False


def _build_warm_restarts_scheduler(
    optimizer: torch.optim.Optimizer,
    T_0: int = 10,
    T_mult: int = 2,
    min_lr: float = 1e-6,
) -> Tuple[torch.optim.lr_scheduler._LRScheduler, bool]:
    """
    Epoch-level cosine warm restarts.
    """
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=min_lr)
    return sched, False


def _build_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 1e-6,
) -> Tuple[torch.optim.lr_scheduler._LRScheduler, bool]:
    """
    Epoch-level ReduceLROnPlateau (call epoch_step(val_metric)).
    """
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience,
                                                       min_lr=min_lr, verbose=True)
    return sched, False


# --------------------------------- Stepper ----------------------------------

class Stepper:
    """
    Unified stepping helper.
      - batch_step(): call every batch (per-step schedulers will step here)
      - epoch_step(metric=None): call at epoch end (plateau/epoch schedulers)
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        step_on_batch: bool = True,
        clip_grad_norm: Optional[float] = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_on_batch = step_on_batch
        self.clip_grad_norm = clip_grad_norm

    def batch_step(self):
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None and self.step_on_batch:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def epoch_step(self, metric: Optional[float] = None):
        if self.scheduler is None:
            return
        # ReduceLROnPlateau needs a metric; others just step
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metric if metric is not None else 0.0)
        elif not self.step_on_batch:
            self.scheduler.step()

    def current_lrs(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# --------------------------- Public entry point -----------------------------

def setup_optimizer_and_scheduler(
    model: nn.Module,
    base_lr: float,
    total_epochs: int,
    # optimizer
    wd: float = 0.05,
    head_lr_mult: float = 5.0,
    head_keywords: Tuple[str, ...] = ("head", "classifier", "fc"),
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    llrd_rules: Optional[List[Tuple[str, float]]] = None,
    # scheduler selection
    variant: str = "warmup_cosine",   # 'warmup_cosine' | 'flat_cosine' | 'warm_restarts' | 'plateau'
    steps_per_epoch: Optional[int] = None,
    warmup_epochs: int = 5,
    min_lr: float = 5e-6,
    warmup_init_factor: float = 0.01,
    flat_ratio: float = 0.3,
    restarts_T0: int = 10,
    restarts_Tmult: int = 2,
    plateau_factor: float = 0.5,
    plateau_patience: int = 3,
    # extras
    clip_grad_norm: Optional[float] = 1.0,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler], Stepper]:
    """
    Returns: (optimizer, scheduler, stepper)
    """
    # 1) Param groups
    groups = build_param_groups(
        model, base_lr=base_lr, wd=wd, head_lr_mult=head_lr_mult,
        head_keywords=head_keywords
    )
    if llrd_rules:
        groups = apply_llrd(model, groups, llrd_rules)

    optimizer = torch.optim.AdamW(groups, betas=betas, eps=eps)

    # 2) Scheduler
    variant = variant.lower()
    scheduler, step_on_batch = None, True

    if variant == "warmup_cosine":
        assert steps_per_epoch is not None, "warmup_cosine requires steps_per_epoch"
        scheduler, step_on_batch = _build_warmup_cosine_scheduler(
            optimizer, total_epochs=total_epochs, steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs, min_lr=min_lr, warmup_init_factor=warmup_init_factor
        )
    elif variant == "flat_cosine":
        scheduler, step_on_batch = _build_flat_cosine_scheduler(
            optimizer, total_epochs=total_epochs, flat_ratio=flat_ratio, min_lr=min_lr
        )
    elif variant == "warm_restarts":
        scheduler, step_on_batch = _build_warm_restarts_scheduler(
            optimizer, T_0=restarts_T0, T_mult=restarts_Tmult, min_lr=min_lr
        )
    elif variant == "plateau":
        scheduler, step_on_batch = _build_plateau_scheduler(
            optimizer, factor=plateau_factor, patience=plateau_patience, min_lr=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler variant: {variant}")

    stepper = Stepper(model, optimizer, scheduler, step_on_batch=step_on_batch, clip_grad_norm=clip_grad_norm)
    return optimizer, scheduler, stepper
