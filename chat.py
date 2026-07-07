# -*- coding: utf-8 -*-
"""
chat.py — инференс и чат с «Lunami-Mini».

Грузит обученный чекпойнт (train.py) + SentencePiece-токенизатор и даёт:
    • интерактивный REPL-чат (ChatML: <|system|>/<|user|>/<|assistant|>/<|end|>);
    • одноразовую генерацию по --prompt;
    • сэмплинг: temperature / top-k / top-p (nucleus) / repetition-penalty (из InferenceConfig).

ВАЖНО про KV-cache: текущая model.py НЕ реализует инкрементальный KV-cache (параметр
use_cache — заглушка: forward всегда прогоняет всю последовательность). Поэтому генерация
идёт ЧЕСТНЫМ full-recompute: на каждый новый токен модель прогоняется по всей текущей
последовательности (обрезанной до max_seq_len). Для 165M-модели и умеренных max_new_tokens
на GPU это приемлемо. Настоящий KV-cache (attention-кэш + рекуррентное состояние Mamba)
потребовал бы доработки model.py — это будущая оптимизация, не входит в v1.

Запуск:
    python chat.py --checkpoint checkpoints/step_50000.pt
    python chat.py --checkpoint checkpoints/step_50000.pt --prompt "Write a bubble sort in Python."
Команды в REPL:  /reset — очистить историю,  /system <текст> — сменить системный промпт,  /exit.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from torch.amp import autocast

from config import InferenceConfig, ModelConfig, SpecialTokens
from model import LunamiLM

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover
    raise SystemExit("ERROR: pip install sentencepiece")

logger = logging.getLogger("chat")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_STOP_IDS = frozenset({SpecialTokens.END, SpecialTokens.EOS})


# ═════════════════════════════════════════════════════════════════════════════════════
# Загрузка модели и токенизатора.
# ═════════════════════════════════════════════════════════════════════════════════════
def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA недоступна — переключаюсь на CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_tokenizer(path: str) -> "spm.SentencePieceProcessor":
    if not os.path.exists(path):
        raise FileNotFoundError(f"Токенизатор не найден: {path} (обучите: python tokenizer_train.py)")
    sp = spm.SentencePieceProcessor(model_file=path)
    for tid, name in SpecialTokens.names().items():
        if sp.piece_to_id(name) != tid:
            raise ValueError(f"Токенизатор рассинхронизирован с config.SpecialTokens ({name!r}).")
    return sp


def load_model(checkpoint_path: str, device: torch.device) -> LunamiLM:
    """Грузит чекпойнт train.py; восстанавливает ModelConfig из сохранённого конфига."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Чекпойнт не найден: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # конфиг модели: из чекпойнта (надёжнее — арх. обязана совпадать с весами), иначе дефолт
    if isinstance(ckpt, dict) and "config" in ckpt and "model" in ckpt.get("config", {}):
        model_cfg = ModelConfig(**ckpt["config"]["model"])
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model_cfg = ModelConfig()
        state = ckpt["model"]
    else:  # голый state_dict
        model_cfg = ModelConfig()
        state = ckpt

    model = LunamiLM(model_cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Отсутствующие ключи при загрузке: %d (напр. %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("Лишние ключи при загрузке: %d (напр. %s)", len(unexpected), unexpected[:3])
    model.to(device).eval()
    logger.info("Модель загружена: %s | контекст=%d | устройство=%s",
                checkpoint_path, model_cfg.max_seq_len, device)
    return model


# ═════════════════════════════════════════════════════════════════════════════════════
# Сэмплинг: repetition penalty + top-k + top-p.
# ═════════════════════════════════════════════════════════════════════════════════════
def apply_repetition_penalty(logits: torch.Tensor, seq_ids: List[int], penalty: float) -> torch.Tensor:
    """Штраф за повтор (как в HF): положительные логиты делим на penalty, отрицательные умножаем."""
    if penalty == 1.0 or not seq_ids:
        return logits
    idx = torch.tensor(sorted(set(seq_ids)), device=logits.device, dtype=torch.long)
    vals = logits[idx]
    logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def filter_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Обнуляет (−inf) хвост распределения по top-k и/или nucleus top-p. logits: (vocab,)."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[1:] = remove[:-1].clone()   # всегда оставляем хотя бы 1 токен
        remove[0] = False
        logits[sorted_idx[remove]] = float("-inf")
    return logits


def sample_next(logits: torch.Tensor, cfg: InferenceConfig) -> int:
    """Возвращает следующий токен: greedy (do_sample=False/temp<=0) или сэмплинг."""
    if not cfg.do_sample or cfg.temperature <= 0.0:
        return int(torch.argmax(logits))
    logits = logits / cfg.temperature
    logits = filter_top_k_top_p(logits, cfg.top_k, cfg.top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))


# ═════════════════════════════════════════════════════════════════════════════════════
# Промпт ChatML.
# ═════════════════════════════════════════════════════════════════════════════════════
_ROLE_TOKEN = {"user": SpecialTokens.USER, "assistant": SpecialTokens.ASSISTANT}


def build_prompt_ids(
    system_prompt: Optional[str],
    history: List[Dict[str, str]],
    sp: "spm.SentencePieceProcessor",
    max_prompt_len: int,
) -> List[int]:
    """
    Собирает ChatML-последовательность, ЗАКАНЧИВАЮЩУЮСЯ маркером <|assistant|>
    (с него модель генерит ответ). Старые реплики отбрасываются, если не влезаем в max_prompt_len.
    """
    head: List[int] = [SpecialTokens.BOS]
    if system_prompt:
        head += [SpecialTokens.SYSTEM] + sp.encode(system_prompt, out_type=int) + [SpecialTokens.END]

    turns: List[List[int]] = []
    for msg in history:
        marker = _ROLE_TOKEN.get(msg["role"])
        if marker is None:
            continue
        turns.append([marker] + sp.encode(msg["content"], out_type=int) + [SpecialTokens.END])

    tail = [SpecialTokens.ASSISTANT]
    budget = max_prompt_len - len(head) - len(tail)
    # набираем реплики с конца (самые свежие), пока влезают
    kept: List[List[int]] = []
    for block in reversed(turns):
        if sum(len(b) for b in kept) + len(block) > budget:
            break
        kept.insert(0, block)

    ids = head
    for block in kept:
        ids += block
    ids += tail
    return ids


# ═════════════════════════════════════════════════════════════════════════════════════
# Генерация (full-recompute).
# ═════════════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate(
    model: LunamiLM,
    sp: "spm.SentencePieceProcessor",
    prompt_ids: List[int],
    cfg: InferenceConfig,
    device: torch.device,
    stream: bool = True,
) -> str:
    """Авторегрессивно генерит ответ, останавливаясь на <|end|>/<|eos|>. Возвращает текст."""
    model.eval()
    amp_dtype = _DTYPE_MAP.get(cfg.dtype, torch.float16)
    amp_enabled = (device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16))
    max_seq = model.cfg.max_seq_len

    ids: List[int] = list(prompt_ids)
    generated: List[int] = []
    printed_len = 0

    for _ in range(cfg.max_new_tokens):
        ctx = ids[-max_seq:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x)["logits"][0, -1, :].float()

        logits = apply_repetition_penalty(logits, ids, cfg.repetition_penalty)
        nxt = sample_next(logits, cfg)

        if nxt in _STOP_IDS:
            break
        ids.append(nxt)
        generated.append(nxt)

        if stream:
            text = sp.decode(generated)
            sys.stdout.write(text[printed_len:])
            sys.stdout.flush()
            printed_len = len(text)

    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return sp.decode(generated)


# ═════════════════════════════════════════════════════════════════════════════════════
# REPL-чат.
# ═════════════════════════════════════════════════════════════════════════════════════
def chat_loop(model: LunamiLM, sp: "spm.SentencePieceProcessor",
              cfg: InferenceConfig, device: torch.device) -> None:
    system_prompt = cfg.system_prompt
    history: List[Dict[str, str]] = []
    max_prompt_len = model.cfg.max_seq_len - cfg.max_new_tokens

    print("=" * 60)
    print("  Lunami-Mini · чат.  /reset — сброс, /system <..> — промпт, /exit — выход")
    print("=" * 60)

    while True:
        try:
            user = input("\n👤 you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            return
        if not user:
            continue
        if user in ("/exit", "/quit"):
            print("Пока!")
            return
        if user == "/reset":
            history.clear()
            print("(история очищена)")
            continue
        if user.startswith("/system"):
            system_prompt = user[len("/system"):].strip() or system_prompt
            history.clear()
            print("(системный промпт обновлён, история очищена)")
            continue

        history.append({"role": "user", "content": user})
        # ограничиваем историю последними max_context_turns репликами (пары user/assistant)
        if len(history) > 2 * cfg.max_context_turns:
            history = history[-2 * cfg.max_context_turns:]

        prompt_ids = build_prompt_ids(system_prompt, history, sp, max_prompt_len)
        sys.stdout.write("🤖 lunami: ")
        sys.stdout.flush()
        answer = generate(model, sp, prompt_ids, cfg, device, stream=True)
        history.append({"role": "assistant", "content": answer})


# ═════════════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    d = InferenceConfig()
    p = argparse.ArgumentParser(description="Чат с Lunami-Mini")
    p.add_argument("--checkpoint", type=str, default=d.checkpoint_path)
    p.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.model")
    p.add_argument("--device", type=str, default=d.device, choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default=d.dtype, choices=["fp16", "bf16", "fp32"])
    p.add_argument("--system", type=str, default=None, help="Системный промпт (иначе из конфига).")
    p.add_argument("--prompt", type=str, default=None, help="Одноразовая генерация вместо REPL.")
    p.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--top-p", type=float, default=d.top_p)
    p.add_argument("--top-k", type=int, default=d.top_k)
    p.add_argument("--repetition-penalty", type=float, default=d.repetition_penalty)
    p.add_argument("--greedy", action="store_true", help="Жадное декодирование (do_sample=False).")
    return p.parse_args()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    args = parse_args()
    cfg = InferenceConfig(
        checkpoint_path=args.checkpoint, tokenizer_path=args.tokenizer, device=args.device,
        dtype=args.dtype, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
        do_sample=not args.greedy,
    )
    if args.system:
        cfg.system_prompt = args.system

    device = resolve_device(cfg.device)
    try:
        sp = load_tokenizer(cfg.tokenizer_path)
        model = load_model(cfg.checkpoint_path, device)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    if args.prompt is not None:
        # одноразовый режим
        history = [{"role": "user", "content": args.prompt}]
        max_prompt_len = model.cfg.max_seq_len - cfg.max_new_tokens
        prompt_ids = build_prompt_ids(cfg.system_prompt, history, sp, max_prompt_len)
        print("🤖 lunami: ", end="", flush=True)
        generate(model, sp, prompt_ids, cfg, device, stream=True)
        return 0

    chat_loop(model, sp, cfg, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
