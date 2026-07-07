# -*- coding: utf-8 -*-
"""
dataset.py — потоковый датасет-пайплайн «Lunami-Mini» (English + Code, v1).

Платформа: Lightning.ai Studios (гибрид T4/A100). Данные СТРИМЯТСЯ с HuggingFace
(streaming=True), поэтому не нужно тянуть терабайты на диск студии.

Что делает:
    1. Грузит обученный SentencePiece-токенизатор (tokenizer/tokenizer.model).
    2. Стримит и СМЕШИВАЕТ по весам источники EN+Code (см. config.default_dataset_sources):
         • pretrain  — сырой текст/код → causal-LM (все токены предсказываем);
         • instruct  — диалоги → чат-формат с LOSS-МАСКОЙ (учим только ответ ассистента).
    3. ПАКУЕТ документы в окна ровно max_seq_len (без PAD-расхода — как в больших LM).
    4. Отдаёт PyTorch IterableDataset → DataLoader (train/val), совместимый с model.py:
         batch = {"input_ids": (B, L), "labels": (B, L)}, где labels с IGNORE_INDEX там,
         где loss считать не нужно. model.py сам делает сдвиг на 1 (next-token).

Чат-формат (ChatML, синхронизирован с config.SpecialTokens):
    <|bos|> <|system|> …sys… <|end|> <|user|> …usr… <|end|> <|assistant|> …resp… <|end|> <|eos|>
    Loss считается ТОЛЬКО на токенах ответа ассистента (+ его <|end|> и финальный <|eos|>),
    остальное маскируется IGNORE_INDEX — модель не учится генерить промпт/вопрос.

Замечания:
    • Пакинг склеивает документы в одном окне (кросс-документное внимание не сбрасывается —
      обычная практика; model.py использует is_causal без блочной маски). Между документами
      стоят <|eos|>/<|bos|>, что даёт модели «мягкий» разделитель.
    • subset/поля датасетов на HF иногда меняются — источники грузятся защищённо, при
      несовпадении даётся понятная ошибка. Для gated (напр. starcoderdata) нужен HF_TOKEN.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# ── SentencePiece (обязателен) ────────────────────────────────────────────────────────
try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover
    raise SystemExit("ERROR: pip install sentencepiece")

from config import (
    DatasetConfig,
    DatasetSource,
    ModelConfig,
    SpecialTokens,
    default_dataset_sources,
    set_seed,
)

logger = logging.getLogger("dataset")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

IGNORE = SpecialTokens.IGNORE_INDEX


# ═════════════════════════════════════════════════════════════════════════════════════
# Токенизатор.
# ═════════════════════════════════════════════════════════════════════════════════════
def load_tokenizer(model_path: str) -> "spm.SentencePieceProcessor":
    """Загружает SentencePiece-модель; при отсутствии — понятная ошибка."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Токенизатор не найден: {model_path}\n"
            f"    Сначала обучите его:  python tokenizer_train.py\n"
            f"    (нужны файлы {model_path} и .vocab рядом)."
        )
    sp = spm.SentencePieceProcessor(model_file=model_path)
    # Санити: строки/ID спецтокенов должны совпадать с config.SpecialTokens.
    for tid, name in SpecialTokens.names().items():
        if sp.piece_to_id(name) != tid:
            raise ValueError(
                f"Токенизатор рассинхронизирован с config.SpecialTokens: "
                f"{name!r} имеет id={sp.piece_to_id(name)}, ожидался {tid}. "
                f"Переобучите токенизатор (python tokenizer_train.py)."
            )
    return sp


# ═════════════════════════════════════════════════════════════════════════════════════
# Инструкции → список сообщений [{role, content}] (роли: system/user/assistant).
# ═════════════════════════════════════════════════════════════════════════════════════
_SHAREGPT_ROLE = {
    "system": "system", "user": "user", "human": "user",
    "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant", "bot": "assistant",
}


def parse_instruct(example: Dict, source: DatasetSource) -> Optional[List[Dict[str, str]]]:
    """
    Приводит один пример instruct-датасета к списку сообщений [{role, content}].
    Поддержаны форматы: "sharegpt" (список {from,value}), "magicoder" (problem/solution),
    "alpaca" (instruction/input/output). Некорректные примеры → None (пропуск).
    """
    fmt = source.instruct_format

    if fmt == "sharegpt":
        conv = example.get(source.conversation_field or "conversations")
        if not isinstance(conv, list) or not conv:
            return None
        messages: List[Dict[str, str]] = []
        for turn in conv:
            if not isinstance(turn, dict):
                return None
            role = _SHAREGPT_ROLE.get(str(turn.get("from", "")).lower())
            content = turn.get("value")
            if role is None or not content:
                continue
            messages.append({"role": role, "content": str(content)})
        return messages or None

    if fmt == "magicoder":
        problem = example.get("problem")
        solution = example.get("solution")
        if not problem or not solution:
            return None
        return [
            {"role": "user", "content": str(problem)},
            {"role": "assistant", "content": str(solution)},
        ]

    if fmt == "alpaca":
        instr = example.get("instruction", "")
        inp = example.get("input", "")
        out = example.get("output", "")
        if not instr or not out:
            return None
        user = instr if not inp else f"{instr}\n\n{inp}"
        return [
            {"role": "user", "content": str(user)},
            {"role": "assistant", "content": str(out)},
        ]

    return None  # неизвестный формат


# ═════════════════════════════════════════════════════════════════════════════════════
# Сообщения → (input_ids, labels) с loss-маской на ответах ассистента.
# ═════════════════════════════════════════════════════════════════════════════════════
_ROLE_TOKEN = {
    "system": SpecialTokens.SYSTEM,
    "user": SpecialTokens.USER,
    "assistant": SpecialTokens.ASSISTANT,
}


def format_chat(
    messages: List[Dict[str, str]],
    sp: "spm.SentencePieceProcessor",
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Собирает ChatML-последовательность и параллельные labels.
    Loss (реальные labels) — только на токенах ответа ассистента + его <|end|>
    и на финальном <|eos|>. Всё остальное = IGNORE. Возвращает None, если в диалоге
    нет ни одного обучаемого токена ассистента.
    """
    if not messages:
        return None

    ids: List[int] = [SpecialTokens.BOS]
    labels: List[int] = [IGNORE]
    trainable = False

    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        marker = _ROLE_TOKEN.get(role)
        if marker is None or not content:
            continue

        content_ids = sp.encode(content, out_type=int)
        # маркер роли — всегда часть промпта (не учим его генерить)
        ids.append(marker)
        labels.append(IGNORE)

        if role == "assistant":
            ids.extend(content_ids)
            labels.extend(content_ids)          # учим генерить ответ
            ids.append(SpecialTokens.END)
            labels.append(SpecialTokens.END)    # ...и его завершение
            trainable = True
        else:
            ids.extend(content_ids)
            labels.extend([IGNORE] * len(content_ids))
            ids.append(SpecialTokens.END)
            labels.append(IGNORE)

    if not trainable:
        return None

    ids.append(SpecialTokens.EOS)
    # учим завершать диалог, только если последняя значимая реплика — ассистента
    last_role = next((m.get("role") for m in reversed(messages)
                      if _ROLE_TOKEN.get(m.get("role")) and (m.get("content") or "").strip()), None)
    labels.append(SpecialTokens.EOS if last_role == "assistant" else IGNORE)

    return ids, labels


# ═════════════════════════════════════════════════════════════════════════════════════
# Pretrain-документ → (input_ids, labels): предсказываем все токены.
# ═════════════════════════════════════════════════════════════════════════════════════
def format_pretrain(
    text: str,
    sp: "spm.SentencePieceProcessor",
) -> Tuple[List[int], List[int]]:
    """[BOS] + токены(text) + [EOS]; labels = input_ids (учим каждый следующий токен)."""
    ids = [SpecialTokens.BOS] + sp.encode(text, out_type=int) + [SpecialTokens.EOS]
    return ids, list(ids)


# ═════════════════════════════════════════════════════════════════════════════════════
# Загрузка одного HF-источника (streaming) + перемешивание буфером.
# ═════════════════════════════════════════════════════════════════════════════════════
def load_hf_source(source: DatasetSource, cfg: DatasetConfig):
    """Грузит HF-датасет в streaming-режиме; при ошибке — понятное сообщение."""
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover
        raise SystemExit("ERROR: pip install datasets")

    token = os.environ.get(cfg.hf_token_env) or None
    try:
        ds = load_dataset(
            source.path,
            source.subset,
            split=source.split,
            streaming=cfg.streaming,
            token=token,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Не удалось загрузить датасет '{source.path}' (subset={source.subset!r}): {e}\n"
            f"    • проверьте название/subset на huggingface.co;\n"
            f"    • для gated-датасетов примите лицензию и задайте env {cfg.hf_token_env};\n"
            f"    • при необходимости поправьте источник в config.default_dataset_sources()."
        ) from e

    if cfg.streaming and cfg.shuffle_buffer > 0:
        ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)
    return ds


def iter_source_docs(
    source: DatasetSource,
    sp: "spm.SentencePieceProcessor",
    cfg: DatasetConfig,
) -> Iterator[Tuple[List[int], List[int]]]:
    """Стримит источник и отдаёт (input_ids, labels) по одному документу/диалогу."""
    ds = load_hf_source(source, cfg)
    warned_schema = False

    for example in ds:
        try:
            if source.kind == "pretrain":
                text = example.get(source.text_field) if isinstance(example, dict) else None
                if not text or len(text) < cfg.min_doc_chars:
                    continue
                ids, labels = format_pretrain(text, sp)
            else:  # instruct
                messages = parse_instruct(example, source)
                if not messages:
                    continue
                out = format_chat(messages, sp)
                if out is None:
                    continue
                ids, labels = out
        except Exception as e:  # noqa: BLE001
            if not warned_schema:
                logger.warning("Пропускаю проблемные примеры источника %s: %s", source.name, e)
                warned_schema = True
            continue

        if ids:
            yield ids, labels


# ═════════════════════════════════════════════════════════════════════════════════════
# Взвешенное смешивание источников.
# ═════════════════════════════════════════════════════════════════════════════════════
def interleave_docs(
    sources: List[DatasetSource],
    sp: "spm.SentencePieceProcessor",
    cfg: DatasetConfig,
    seed: int,
) -> Iterator[Tuple[List[int], List[int]]]:
    """
    Случайно выбирает источник по нормированному weight и отдаёт его следующий документ.
    Исчерпавшиеся источники выбывают. Порядок детерминирован seed'ом (воспроизводимость).
    """
    rng = random.Random(seed)
    iterators = [iter_source_docs(s, sp, cfg) for s in sources]
    weights = [max(s.weight, 0.0) for s in sources]
    total = sum(weights) or 1.0
    probs = [w / total for w in weights]
    active = [i for i, w in enumerate(weights) if w > 0.0] or list(range(len(iterators)))

    while active:
        pick = rng.choices(active, weights=[probs[i] for i in active], k=1)[0]
        try:
            yield next(iterators[pick])
        except StopIteration:
            active.remove(pick)


# ═════════════════════════════════════════════════════════════════════════════════════
# Пакинг документов в окна ровно block_size.
# ═════════════════════════════════════════════════════════════════════════════════════
def pack_stream(
    doc_iter: Iterator[Tuple[List[int], List[int]]],
    block_size: int,
) -> Iterator[Tuple[List[int], List[int]]]:
    """
    Склеивает (input_ids, labels) документов и режет на окна ровно block_size токенов.
    Хвост короче block_size отбрасывается (стандартно для packing → однородные окна).
    """
    buf_ids: List[int] = []
    buf_lab: List[int] = []
    for ids, labels in doc_iter:
        buf_ids.extend(ids)
        buf_lab.extend(labels)
        while len(buf_ids) >= block_size:
            yield buf_ids[:block_size], buf_lab[:block_size]
            del buf_ids[:block_size]
            del buf_lab[:block_size]


# ═════════════════════════════════════════════════════════════════════════════════════
# PyTorch IterableDataset.
# ═════════════════════════════════════════════════════════════════════════════════════
class LunamiIterableDataset(IterableDataset):
    """
    Потоковый датасет: смешивание → пакинг → окна (input_ids, labels) длиной max_seq_len.

    split="val" отдаёт первые val_take окон, split="train" — все остальные (детерминированно
    по seed, поэтому train/val не пересекаются). Многопроцессные воркеры шардятся по индексу
    окна (worker_id), чтобы не дублировать данные. Для streaming рекомендуется num_workers=0..2.
    """

    def __init__(
        self,
        data_cfg: DatasetConfig,
        model_cfg: ModelConfig,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        super().__init__()
        assert split in ("train", "val")
        self.data_cfg = data_cfg
        self.block_size = model_cfg.max_seq_len
        self.split = split
        self.seed = seed
        self.val_take = data_cfg.val_take

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Токенизатор грузим ВНУТРИ воркера (SentencePieceProcessor не всегда пиклится).
        sp = load_tokenizer(self.data_cfg.tokenizer_model_path)
        sources = self.data_cfg.sources or default_dataset_sources()

        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0

        docs = interleave_docs(sources, sp, self.data_cfg, self.seed)
        blocks = pack_stream(docs, self.block_size)

        for gidx, (ids, labels) in enumerate(blocks):
            # train/val split по глобальному индексу окна
            if self.split == "val":
                if gidx >= self.val_take:
                    break
            else:  # train
                if gidx < self.val_take:
                    continue
            # шардинг по воркерам
            if (gidx % num_workers) != worker_id:
                continue
            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }


# ═════════════════════════════════════════════════════════════════════════════════════
# collate + DataLoaders.
# ═════════════════════════════════════════════════════════════════════════════════════
def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Все окна одинаковой длины (block_size) → простой stack, без паддинга."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def build_dataloaders(
    model_cfg: ModelConfig,
    data_cfg: DatasetConfig,
    micro_batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Возвращает (train_loader, val_loader). Вызывается из train.py.
    Батч: {"input_ids": (B, L), "labels": (B, L)} — прямо в model(input_ids, labels=...).
    """
    if not data_cfg.sources:
        data_cfg.sources = default_dataset_sources()
    set_seed(seed)

    train_ds = LunamiIterableDataset(data_cfg, model_cfg, split="train", seed=seed)
    val_ds = LunamiIterableDataset(data_cfg, model_cfg, split="val", seed=seed)

    common = dict(
        batch_size=micro_batch_size,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if num_workers > 0:
        common["persistent_workers"] = True
    train_loader = DataLoader(train_ds, **common)
    val_loader = DataLoader(val_ds, **common)
    return train_loader, val_loader


# ═════════════════════════════════════════════════════════════════════════════════════
# Smoke-test (реальные датасеты; требует сети и `pip install datasets`).
# ═════════════════════════════════════════════════════════════════════════════════════
def main() -> int:
    for stream in (__import__("sys").stdout, __import__("sys").stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    model_cfg = ModelConfig()
    data_cfg = DatasetConfig()
    data_cfg.sources = default_dataset_sources()
    data_cfg.val_take = 4          # крошечный val для быстрого прогона
    data_cfg.shuffle_buffer = 100

    logger.info("Smoke-test dataset.py: строю DataLoader и тяну 1 батч (нужна сеть + HF)...")
    try:
        sp = load_tokenizer(data_cfg.tokenizer_model_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    try:
        train_loader, _ = build_dataloaders(
            model_cfg, data_cfg, micro_batch_size=2, num_workers=0, seed=42
        )
        batch = next(iter(train_loader))
    except Exception as e:  # noqa: BLE001
        logger.error("Не удалось получить батч (сеть/датасеты?): %s", e)
        return 1

    ids = batch["input_ids"]
    labels = batch["labels"]
    trainable = int((labels != IGNORE).sum())
    logger.info("input_ids: %s | labels: %s", tuple(ids.shape), tuple(labels.shape))
    logger.info("обучаемых токенов в батче: %d / %d", trainable, labels.numel())
    logger.info("пример декодинга первых 120 токенов:\n%s",
                sp.decode(ids[0, :120].tolist()))
    logger.info("✅ dataset.py работает.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
