# -*- coding: utf-8 -*-
"""
train.py — тренировочный цикл «Lunami-Mini» на Lightning.ai Studios (гибрид T4/A100).

Связывает всё воедино:
    config.py  — гиперпараметры + профили железа (Ракета/Рабочая лошадка);
    model.py   — LunamiLM (Mamba-2 + GQA + MoE), forward → {logits, loss, aux_loss};
    dataset.py — потоковые DataLoader'ы (packing, chat loss-маска).

Возможности:
    • Выбор профиля железа: --profile rocket (A100 bf16 + torch.compile) | workhorse (T4 fp16).
    • AMP: bf16 (без GradScaler) или fp16 (с динамическим GradScaler).
    • Градиентная аккумуляция (effective batch из профиля), клиппинг нормы градиента.
    • Cosine LR с линейным warmup и полом min_lr_ratio.
    • torch.compile() (профиль Ракета), gradient checkpointing (профиль Лошадка).
    • Чекпойнты (model/opt/sched/scaler/step) + ротация keep_last_n + бэкап в Teamspace.
    • Периодический eval на val (loss + perplexity), логирование в консоль и wandb.
    • Возобновление обучения: --resume <path>.

Запуск (в Lightning Studio):
    A100:  python train.py --profile rocket
    T4:    python train.py --profile workhorse
    resume: python train.py --profile rocket --resume checkpoints/step_5000.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from config import (
    DatasetConfig,
    LunamiConfig,
    ModelConfig,
    TrainConfig,
    apply_hardware_profile,
    count_parameters,
    default_dataset_sources,
    human_readable,
    set_seed,
)
from dataset import build_dataloaders
from model import LunamiLM

logger = logging.getLogger("train")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


# ═════════════════════════════════════════════════════════════════════════════════════
# Аргументы командной строки.
# ═════════════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение Lunami-Mini")
    p.add_argument("--profile", choices=["rocket", "workhorse"], default=None,
                   help="Профиль железа: rocket=A100/bf16/compile, workhorse=T4/fp16. "
                        "По умолчанию — из TrainConfig.hardware_profile.")
    p.add_argument("--resume", type=str, default=None, help="Путь к чекпойнту для догонки.")
    p.add_argument("--max-steps", type=int, default=None, help="Переопределить total_steps.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. Для streaming рекомендуется 0 (иначе повторный "
                        "стриминг на каждого воркера).")
    p.add_argument("--no-wandb", action="store_true", help="Отключить wandb даже если включён в конфиге.")
    p.add_argument("--compile", dest="compile", action="store_true", default=None,
                   help="Принудительно включить torch.compile().")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="Принудительно выключить torch.compile().")
    p.add_argument("--backup-dir", type=str, default=None,
                   help="Копировать каждый чекпойнт сюда (напр. /content/drive/MyDrive/... "
                        "в Colab — сессия там не персистентна, чекпойнты в output_dir пропадут "
                        "при отключении рантайма).")
    p.add_argument("--micro-batch-size", type=int, default=None,
                   help="Переопределить micro_batch_size профиля (уменьшить при OOM).")
    p.add_argument("--grad-accum-steps", type=int, default=None,
                   help="Переопределить grad_accum_steps профиля (увеличить, если уменьшили "
                        "micro-batch-size, чтобы сохранить тот же effective batch).")
    p.add_argument("--shuffle-buffer", type=int, default=None,
                   help="Переопределить DatasetConfig.shuffle_buffer (по умолчанию 10000 — "
                        "на некоторых огромных HF-шардах (codeparrot-clean-train) первое "
                        "заполнение буфера может зависать надолго; уменьшите до 100-500 "
                        "для быстрого старта ценой менее качественного перемешивания).")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="Переопределить ModelConfig.max_seq_len (по умолчанию 8192 — не влезает "
                        "на T4 даже при micro_batch_size=1, см. roadmap.md#compute). Все "
                        "чекпойнты этого проекта на T4 обучались с 2048 — используйте это "
                        "значение при --profile workhorse, если не меняли профиль вручную.")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════════════
# Устройство и точность.
# ═════════════════════════════════════════════════════════════════════════════════════
def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
        return dev
    logger.warning("CUDA недоступна — обучение на CPU (только для отладки, крайне медленно).")
    return torch.device("cpu")


# ═════════════════════════════════════════════════════════════════════════════════════
# Оптимизатор: AdamW с раздельными группами weight decay.
# ═════════════════════════════════════════════════════════════════════════════════════
def build_optimizer(model: nn.Module, cfg: TrainConfig, device: torch.device) -> torch.optim.Optimizer:
    """
    Weight decay применяем только к матрицам (ndim>=2): линейные слои, эмбеддинги.
    Нормы (RMSNorm), bias и прочие 1D-параметры — без decay (стандарт nanoGPT/Llama).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused AdamW заметно быстрее на CUDA.
    use_fused = (device.type == "cuda")
    try:
        opt = torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=cfg.betas,
                                eps=cfg.epsilon, fused=use_fused)
    except (RuntimeError, TypeError):
        opt = torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=cfg.betas, eps=cfg.epsilon)
    logger.info("Оптимизатор AdamW | decay-параметров: %d | без decay: %d | fused=%s",
                len(decay), len(no_decay), use_fused)
    return opt


# ═════════════════════════════════════════════════════════════════════════════════════
# Расписание LR: линейный warmup → cosine до пола min_lr_ratio.
# ═════════════════════════════════════════════════════════════════════════════════════
def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> torch.optim.lr_scheduler.LRScheduler:
    warmup = max(1, cfg.warmup_steps)
    total = max(warmup + 1, cfg.total_steps)
    floor = cfg.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:                       # линейный разогрев 0→1
            return step / warmup
        if step >= total:                       # после конца — держим пол
            return floor
        progress = (step - warmup) / (total - warmup)   # 0→1
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1→0
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═════════════════════════════════════════════════════════════════════════════════════
# Бесконечный поток батчей (streaming DataLoader может «закончиться» → переинициализируем).
# ═════════════════════════════════════════════════════════════════════════════════════
def cycle(loader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


# ═════════════════════════════════════════════════════════════════════════════════════
# Чекпойнты.
# ═════════════════════════════════════════════════════════════════════════════════════
def save_checkpoint(
    path: Path,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    step: int,
    full_cfg: LunamiConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": full_cfg.to_dict(),
    }, path)


def rotate_checkpoints(output_dir: Path, keep_last_n: int) -> None:
    """Оставляем только keep_last_n последних step_*.pt (по номеру шага)."""
    ckpts = sorted(output_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for old in ckpts[:-keep_last_n] if keep_last_n > 0 else []:
        try:
            old.unlink()
        except OSError:
            pass


def maybe_backup(path: Path, backup_dir: Optional[str]) -> None:
    if not backup_dir:
        return
    try:
        dst = Path(backup_dir)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst / path.name)
    except OSError as e:
        logger.warning("Не удалось скопировать чекпойнт в бэкап %s: %s", backup_dir, e)


# ═════════════════════════════════════════════════════════════════════════════════════
# Оценка на валидации.
# ═════════════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    eval_steps: int,
) -> Dict[str, float]:
    """Средний CE-loss (без MoE aux) и perplexity по eval_steps батчам val."""
    model.eval()
    losses: List[float] = []
    it = iter(val_loader)
    for _ in range(eval_steps):
        try:
            batch = next(it)
        except StopIteration:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(input_ids, labels=labels)
        # чистый CE = loss - aux (для честной perplexity)
        ce = (out["loss"] - out["aux_loss"]).float().item()
        losses.append(ce)
    model.train()
    if not losses:
        return {"val_loss": float("nan"), "val_ppl": float("nan")}
    avg = sum(losses) / len(losses)
    return {"val_loss": avg, "val_ppl": math.exp(min(avg, 20.0))}


# ═════════════════════════════════════════════════════════════════════════════════════
# Главный цикл обучения.
# ═════════════════════════════════════════════════════════════════════════════════════
def train(full_cfg: LunamiConfig, args: argparse.Namespace) -> None:
    mcfg: ModelConfig = full_cfg.model
    tcfg: TrainConfig = full_cfg.train
    dcfg: DatasetConfig = full_cfg.dataset

    device = get_device()
    set_seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(tcfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True   # ускорение matmul на A100
        torch.backends.cudnn.allow_tf32 = True

    # ── точность / AMP ────────────────────────────────────────────────────────
    amp_dtype = _DTYPE_MAP.get(tcfg.dtype, torch.float16)
    amp_enabled = (device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16))
    scaler_enabled = (device.type == "cuda" and amp_dtype == torch.float16 and tcfg.dynamic_loss_scale)
    scaler = GradScaler(device=device.type, enabled=scaler_enabled)
    logger.info("Профиль=%s | dtype=%s | AMP=%s | GradScaler=%s | eff.batch=%d (micro=%d×accum=%d)",
                tcfg.hardware_profile, tcfg.dtype, amp_enabled, scaler_enabled,
                tcfg.effective_batch_size, tcfg.micro_batch_size, tcfg.grad_accum_steps)

    # ── модель ────────────────────────────────────────────────────────────────
    model = LunamiLM(mcfg).to(device)
    logger.info("%s", model.summary())
    params = count_parameters(mcfg)
    logger.info("Активных параметров/токен ≈ %s | всего ≈ %s",
                human_readable(params["ACTIVE_per_token"]), human_readable(params["TOTAL_all"]))

    if tcfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: ВКЛ (экономия памяти).")

    raw_model = model  # ссылка на некомпилированную модель для state_dict/сохранения
    if tcfg.use_torch_compile:
        logger.info("torch.compile(mode=%s)... (первый шаг будет дольше)", tcfg.torch_compile_mode)
        model = torch.compile(model, mode=tcfg.torch_compile_mode)

    # ── данные ────────────────────────────────────────────────────────────────
    if not dcfg.sources:
        dcfg.sources = default_dataset_sources()
    train_loader, val_loader = build_dataloaders(
        mcfg, dcfg, micro_batch_size=tcfg.micro_batch_size,
        num_workers=args.num_workers, seed=tcfg.seed,
    )

    # ── оптимизатор / расписание ──────────────────────────────────────────────
    optimizer = build_optimizer(raw_model, tcfg, device)
    scheduler = build_scheduler(optimizer, tcfg)

    # ── wandb (опционально) ───────────────────────────────────────────────────
    wandb_run = None
    if tcfg.use_wandb and not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=tcfg.wandb_project, entity=tcfg.wandb_entity,
                                   config=full_cfg.to_dict())
        except Exception as e:  # noqa: BLE001
            logger.warning("wandb не инициализирован (%s) — продолжаю без него.", e)

    # ── возобновление ─────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        logger.info("Возобновление из %s ...", args.resume)
        # map_location="cpu": модель/оптимизатор уже созданы на GPU, поэтому чекпойнт
        # грузим на CPU и копируем через load_state_dict — иначе на секунду в памяти
        # GPU оказываются две полные копии весов+состояния Adam разом (риск CUDA OOM
        # именно при возобновлении, когда свободной памяти и так впритык на T4).
        # weights_only=False: это НАШ доверенный чекпойнт (содержит конфиг-словарь, не только веса).
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt["step"]) + 1
        del ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Возобновлено со шага %d.", start_step)

    # ── сохраняем конфиг рядом с чекпойнтами ──────────────────────────────────
    out_dir = Path(tcfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_cfg.save(out_dir / "config.json")

    # ── ЦИКЛ ──────────────────────────────────────────────────────────────────
    total_steps = tcfg.total_steps
    data_iter = cycle(train_loader)
    model.train()
    tokens_per_step = tcfg.effective_batch_size * mcfg.max_seq_len
    t0 = time.time()
    running_loss = 0.0
    saved_ckpts: List[Path] = []

    logger.info("═" * 70)
    logger.info("СТАРТ ОБУЧЕНИЯ: шагов %d → %d, токенов/шаг=%s",
                start_step, total_steps, f"{tokens_per_step:,}")
    logger.info("═" * 70)

    for step in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_aux = 0.0

        # ── градиентная аккумуляция ──
        for _ in range(tcfg.grad_accum_steps):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(input_ids, labels=labels)
                loss = out["loss"] / tcfg.grad_accum_steps
            scaler.scale(loss).backward()
            step_loss += loss.item()
            step_aux += out["aux_loss"].item() / tcfg.grad_accum_steps

        # ── клиппинг + шаг оптимизатора ──
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tcfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += step_loss

        # ── логирование ──
        if (step + 1) % tcfg.log_interval == 0:
            dt = time.time() - t0
            steps_done = (step + 1) - start_step
            tok_per_sec = tokens_per_step * steps_done / max(dt, 1e-6)
            avg_loss = running_loss / tcfg.log_interval
            lr_now = scheduler.get_last_lr()[0]
            logger.info(
                "step %6d/%d | loss %.4f | aux %.4f | lr %.2e | gnorm %.2f | %s tok/s",
                step + 1, total_steps, avg_loss, step_aux, lr_now, float(grad_norm),
                f"{tok_per_sec:,.0f}",
            )
            if wandb_run:
                wandb_run.log({"train/loss": avg_loss, "train/aux": step_aux,
                               "train/lr": lr_now, "train/grad_norm": float(grad_norm),
                               "perf/tokens_per_sec": tok_per_sec}, step=step + 1)
            running_loss = 0.0

        # ── eval ──
        if (step + 1) % tcfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, device, amp_dtype, amp_enabled, tcfg.eval_steps)
            logger.info("── EVAL step %d | val_loss %.4f | val_ppl %.2f",
                        step + 1, metrics["val_loss"], metrics["val_ppl"])
            if wandb_run:
                wandb_run.log(metrics, step=step + 1)

        # ── чекпойнт ──
        if (step + 1) % tcfg.save_every_steps == 0:
            ckpt_path = out_dir / f"step_{step + 1}.pt"
            save_checkpoint(ckpt_path, raw_model, optimizer, scheduler, scaler, step, full_cfg)
            rotate_checkpoints(out_dir, tcfg.keep_last_n)
            maybe_backup(ckpt_path, tcfg.backup_dir)
            logger.info("💾 Чекпойнт сохранён: %s", ckpt_path)

    # ── финальный чекпойнт ──
    final_path = out_dir / f"step_{total_steps}.pt"
    save_checkpoint(final_path, raw_model, optimizer, scheduler, scaler, total_steps - 1, full_cfg)
    maybe_backup(final_path, tcfg.backup_dir)
    logger.info("✅ Обучение завершено. Финальный чекпойнт: %s", final_path)
    if wandb_run:
        wandb_run.finish()


# ═════════════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════════════
def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    args = parse_args()

    # Собираем конфиг и применяем профиль железа.
    full_cfg = LunamiConfig()
    full_cfg.dataset.sources = default_dataset_sources()
    profile = args.profile or full_cfg.train.hardware_profile
    apply_hardware_profile(full_cfg.train, profile)

    # CLI-переопределения.
    if args.max_steps is not None:
        full_cfg.train.total_steps = args.max_steps
    if args.compile is not None:
        full_cfg.train.use_torch_compile = args.compile
    if args.backup_dir is not None:
        full_cfg.train.backup_dir = args.backup_dir
    if args.shuffle_buffer is not None:
        full_cfg.dataset.shuffle_buffer = args.shuffle_buffer
    if args.max_seq_len is not None:
        full_cfg.model.max_seq_len = args.max_seq_len
    if args.micro_batch_size is not None:
        full_cfg.train.micro_batch_size = args.micro_batch_size
    if args.grad_accum_steps is not None:
        full_cfg.train.grad_accum_steps = args.grad_accum_steps
    if args.micro_batch_size is not None or args.grad_accum_steps is not None:
        full_cfg.train.__post_init__()  # пересчитать effective_batch_size под новые micro/accum

    train(full_cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
