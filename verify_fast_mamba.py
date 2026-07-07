# -*- coding: utf-8 -*-
"""
verify_fast_mamba.py — проверка быстрого бэкенда Mamba-2 (mamba_ssm) перед реальным обучением.

ВАЖНО: быстрый бэкенд в model.py (mamba_ssm.modules.mamba2.Mamba2) не тестировался
автором локально — нет подходящего CUDA-окружения для компиляции mamba_ssm (Windows).
Прежде чем доверять ему в реальном обучении — прогоните этот скрипт на машине с GPU,
где mamba_ssm реально устанавливается (Lightning.ai / Colab, Linux + CUDA).

Запуск:
    pip install mamba-ssm causal-conv1d
    python verify_fast_mamba.py

Проверяет:
    1. mamba_ssm реально импортируется, и Mamba2Block реально создаёт fast_backend.
    2. Forward/backward не падает, logits/loss/градиенты конечны (нет NaN/Inf).
    3. Несколько реальных шагов обучения — loss убывает (не гарантия корректности
       на случайном шуме, но ловит грубые поломки).
    4. Грубое сравнение скорости fast vs референсная PyTorch-реализация —
       ради этого всё и затевалось, здесь должно быть заметное ускорение.

Если что-то из этого не проходит — НЕ используйте mamba_use_fast=True для реального
обучения, пока не разберётесь (см. предупреждения, которые печатает model.py при
неудачной инициализации fast_backend — они укажут на конкретную причину).
"""
from __future__ import annotations

import time

import torch

import model as M
from config import ModelConfig


def build_model(use_fast: bool, seed: int = 0):
    torch.manual_seed(seed)
    cfg = ModelConfig(
        d_model=256, max_seq_len=512, vocab_size=1024,
        n_layers=8, n_mamba=6, n_attention=2,
        layer_pattern=(0, 0, 0, 1, 0, 0, 0, 1),
        n_heads_q=4, n_kv_heads=2, n_experts=4, n_active_experts=2, ffn_expand=2,
        mamba_use_fast=use_fast,
    )
    return M.LunamiLM(cfg), cfg


def main() -> int:
    if not M.MAMBA_SSM_AVAILABLE:
        print("ERROR: mamba_ssm не установлен/не импортируется.")
        print("Установите:  pip install mamba-ssm causal-conv1d")
        print("и запустите заново на машине с GPU (CUDA-компиляция обязательна).")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: mamba_ssm fused-кернели требуют CUDA GPU, а его нет в этом окружении.")
        return 1

    print("=" * 70)
    print("1) fast_backend реально создаётся в Mamba2Block")
    fast_model, cfg = build_model(use_fast=True)
    fast_model.to(device)
    n_with_fast_backend = sum(
        1 for layer in fast_model.layers
        if hasattr(layer, "block") and getattr(layer.block, "fast_backend", None) is not None
    )
    print(f"   Mamba-слоёв с fast_backend: {n_with_fast_backend} / {cfg.n_mamba}")
    if n_with_fast_backend != cfg.n_mamba:
        print("   ОШИБКА: fast_backend создался не во всех Mamba-слоях (смотрите "
              "warnings выше — конструктор mamba_ssm.Mamba2 упал, был fallback).")
        return 1
    print("   OK")

    print("=" * 70)
    print("2) Forward/backward: выход конечен, backward не падает")
    ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device=device)
    out = fast_model(ids, labels=ids)
    assert torch.isfinite(out["logits"]).all(), "logits содержат NaN/Inf!"
    assert torch.isfinite(out["loss"]), "loss не конечен!"
    out["loss"].backward()
    for name, p in fast_model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            print(f"   ОШИБКА: градиент {name} содержит NaN/Inf!")
            return 1
    print(f"   OK (loss={out['loss'].item():.4f}, все градиенты конечны)")

    print("=" * 70)
    print("3) Реальное обучение: loss должен убывать за несколько шагов")
    fast_model, cfg = build_model(use_fast=True)
    fast_model.to(device)
    opt = torch.optim.AdamW(fast_model.parameters(), lr=1e-3)
    losses = []
    for _ in range(10):
        ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device=device)
        opt.zero_grad()
        out = fast_model(ids, labels=ids)
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    print(f"   loss по шагам: {[round(l, 3) for l in losses]}")
    if losses[-1] >= losses[0]:
        print("   ПРЕДУПРЕЖДЕНИЕ: loss не убыл за 10 шагов (на случайном шуме не всегда "
              "показательно — для полной уверенности смотрите реальный train.py).")
    else:
        print("   OK (loss убывает)")

    print("=" * 70)
    print("4) Грубое сравнение скорости: fast backend vs референсная реализация")
    slow_model, _ = build_model(use_fast=False)
    slow_model.to(device)

    def timed_steps(m: torch.nn.Module, n: int = 5) -> float:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len), device=device)
            out = m(ids, labels=ids)
            out["loss"].backward()
            m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return (time.time() - t0) / n

    timed_steps(fast_model, n=2)   # прогрев (CUDA-кернели компилируются при первом вызове)
    timed_steps(slow_model, n=2)
    t_fast = timed_steps(fast_model)
    t_slow = timed_steps(slow_model)
    print(f"   fast backend       : {t_fast * 1000:.1f} мс/шаг")
    print(f"   референс (PyTorch) : {t_slow * 1000:.1f} мс/шаг")
    print(f"   ускорение          : {t_slow / t_fast:.2f}x")

    print("=" * 70)
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ. Быстрый бэкенд можно использовать для реального обучения.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
