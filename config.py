# -*- coding: utf-8 -*-
"""
config.py — центральная конфигурация модели "Lunami-Mini".

Архитектура 2026 года: гибрид Mamba-2 SSM + GQA-Transformer + MoE (Mixture of Experts).

    Вся конфигурация собрана в одном месте, чтобы model.py / dataset.py /
    train.py / chat.py / tokenizer_train.py ссылались на единый источник истины
    и при изменении гиперпараметров не приходилось править несколько файлов.

Философия:
    - d_model = 768, 24 слоя, контекст 8192 токенов.
    - 75 % слоёв — Mamba-2 (быстрый, эффективный SSM), 25 % — GQA Attention (точность).
    - Чередование жёстко зафиксировано: [M, M, M, A] × 6 = 24 слоя.
    - FFN везде выполнен как MoE (8 экспертов, top-2 роутинг) со SwiGLU-активацией.
    - Нормализация — RMSNorm, эмбеддинги привязаны (tie), позиционная кодировка — RoPE.
    - Токенизатор: SentencePiece BPE, словарь 65536, EN+Code (v1, без русского).

    Параметр "~125M" из ТЗ трактуется как число АКТИВНЫХ параметров на токен
    (tied эмбеддинги + 2 активных эксперта из 8). Полное число хранящихся параметров
    выше из-за MoE — это нормально для MoE-моделей (Mixtral и др.). Точную оценку
    см. в функции count_parameters() в конце файла.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Специальные токены — ID зафиксированы вручную, чтобы BPE-обучение и инференс
# всегда интерпретировали их одинаково, независимо от частоты в корпусе.
# ─────────────────────────────────────────────────────────────────────────────
class SpecialTokens:
    """Зарезервированные специальные токены и их фиксированные ID."""

    # Раскладка ID синхронизирована с тем, как их резервирует SentencePiece:
    #   id 0..3 — встроенные спецтокены (pad/bos/eos/unk) сплошным блоком в начале словаря;
    #   id 4..9 — структурные токены (chat/code) через user_defined_symbols, строго по порядку.
    PAD: int = 0          # паддинг (игнорируется в loss через ignore_index)
    BOS: int = 1          # начало последовательности
    EOS: int = 2          # конец последовательности (модель учится его генерить)
    UNK: int = 3          # неизвестный токен (нужен SentencePiece при byte_fallback; на практике почти не эмиттится)
    SYSTEM: int = 4       # <|system|> — маркер начала системной реплики
    USER: int = 5         # <|user|>   — маркер начала реплики пользователя
    ASSISTANT: int = 6    # <|assistant|> — маркер начала ответа модели (по нему считаем loss)
    END: int = 7          # <|end|>    — общий терминатор реплики (ChatML-style, один на все роли)
    CODE: int = 8         # <|code|> ... <|endofcode|>
    ENDOFCODE: int = 9
    IGNORE_INDEX: int = -100  # маска для loss: токены user/system не учим предсказывать

    # Человекочитаемые имена для BPE-токенизатора (полезно при сохранении/загрузке).
    NAMES: Dict[int, str] = field(default_factory=dict)

    @classmethod
    def names(cls) -> Dict[int, str]:
        """Возвращает словарь {id: строковое имя} для всех специальных токенов.

        Все строки уникальны. Раньше SYSTEM_END/USER_END/ASSISTANT_END совпадали как
        «<|end|>» — это ломало токенизатор (одна строка не может иметь три разных ID).
        Теперь один общий терминатор реплики <|end|> (ChatML-style).
        """
        return {
            cls.PAD: "<|pad|>",
            cls.BOS: "<|bos|>",
            cls.EOS: "<|eos|>",
            cls.UNK: "<|unk|>",
            cls.SYSTEM: "<|system|>",
            cls.USER: "<|user|>",
            cls.ASSISTANT: "<|assistant|>",
            cls.END: "<|end|>",
            cls.CODE: "<|code|>",
            cls.ENDOFCODE: "<|endofcode|>",
        }

    @classmethod
    def all_strings(cls) -> List[str]:
        """Все строковые формы специальных токенов (для резервирования в BPE)."""
        return list(cls.names().values())

    @classmethod
    def parse(cls, s: str) -> Optional[int]:
        """Если строка совпадает с одним из специальных токенов — вернёт его ID."""
        for tid, name in cls.names().items():
            if s == name:
                return tid
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Главная конфигурация.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Гиперпараметры архитектуры модели Lunami-Mini."""

    # ── Общие размеры ──────────────────────────────────────────────────────
    vocab_size: int = 65_536          # размер словаря BPE (EN+Code, v1)
    d_model: int = 768                # ширина модели (размер скрытого состояния)
    max_seq_len: int = 8_192          # длина контекста в токенах
    tie_embeddings: bool = True       # привязать embedding и lm_head (экономия параметров)

    # ── Слои ────────────────────────────────────────────────────────────────
    n_layers: int = 24                # всего слоёв
    n_mamba: int = 18                 # 75 % — Mamba-2
    n_attention: int = 6              # 25 % — Attention
    # Шаблон чередования: индекс типа блока на каждую позицию. 0 = Mamba, 1 = Attention.
    # [M, M, M, A] × 6 → 24 слоя.
    layer_pattern: Tuple[int, ...] = (
        0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 1,
    )  # type: ignore[assignment]

    # ── Mamba-2 блок ────────────────────────────────────────────────────────
    mamba_d_state: int = 64           # размер скрытого состояния SSM (N в Mamba-2)
    mamba_expand: int = 2             # фактор расширения внутреннего канала: d_inner = expand*d_model
    mamba_conv_kernel: int = 4        # ядро короткой conv1d (как в оригинальном Mamba)
    mamba_chunk_size: int = 256        # размер чанка для chunkwise-вычисления (Mamba-2/ssd)
    mamba_use_fast: bool = True        # использовать качественную быструю реализацию, если доступна

    # ── Transformer-Attention блок (GQA как в Llama 3) ───────────────────────
    n_heads_q: int = 8                # число query heads
    n_kv_heads: int = 2               # число key/value heads (GQA: kv меньше q)
    head_dim: int = 96                 # d_model // n_heads_q = 768 // 8
    rope_theta: float = 10_000.0      # базовая частота RoPE
    rope_scaling: Optional[float] = None  # None — без скейлинга; float — линейное растяжение
    flash_attention: bool = True     # пытаться использовать FlashAttention-2 если доступно

    # ── MoE (Mixture of Experts) FFN ────────────────────────────────────────
    use_moe: bool = True              # True — MoE в FFN, False — обычный плотный SwiGLU
    n_experts: int = 8                # всего экспертов
    n_active_experts: int = 2         # активных экспертов на токен (top-2 роутинг)
    moe_router_init_std: float = 1.0  # std инициализации линейного роутера, масштабированный на d_model
    moe_router_bias: bool = False     # bias у роутера
    moe_aux_loss_weight: float = 0.01 # вес load-balancing (вспомогательного) лосса
    moe_aux_loss_free_routing: bool = False  # True — детерминированный top-1 + вспом. (как в Switch--trafo по желанию)
    moe_noise_std: float = 0.0        # шум в роутере при обучении (0 — без шума)
    moe_grouped_gemm: bool = False    # использовать GroupedGEMM, если поддерживается (ускорение MoE)

    # ── FFN (плотный путь если MoE выключен, или внутренности эксперта) ──────
    ffn_expand: int = 2               # скрытая размерность: int(d_model * 2 * 2/3) ≈ SwiGLU-стиль
    activation: str = "swiglu"        # SwiGLU, как в Llama

    # ── Нормализация и стабилизация ─────────────────────────────────────────
    norm_eps: float = 1e-6            # epsilon для RMSNorm
    use_bias: bool = False            # bias в линейных слоях — как в Llama, ставим False
    dropout: float = 0.0               # dropout (0 на практике для small LM, оставлен как рычаг)
    initializer_range: float = 0.02   # std обычных слоёв инициализации

    # ── Инициализация SSM специальных параметров ─────────────────────────────
    mamba_init_A: str = "s4d-linspace"  # вид инициализации A_log (как в Mamba: s4d-linspace/normal)
    mamba_init_D: float = 1.0

    # ── Чтобы Pydantic-style strict не ругался на кортеж — конструктор-проверка ─
    def __post_init__(self) -> None:
        # Проверяем согласованность числа слоёв и шаблона.
        if len(self.layer_pattern) != self.n_layers:
            raise ValueError(
                f"layer_pattern длиной {len(self.layer_pattern)} != n_layers {self.n_layers}"
            )
        n_m = sum(1 for x in self.layer_pattern if x == 0)
        n_a = sum(1 for x in self.layer_pattern if x == 1)
        if n_m != self.n_mamba or n_a != self.n_attention:
            raise ValueError(
                f"Несовпадение числа блоков: pattern даёт M={n_m}, A={n_a}, "
                f"а конфиг ожидает M={self.n_mamba}, A={self.n_attention}"
            )
        if self.d_model % self.n_heads_q != 0:
            raise ValueError(
                f"d_model ({self.d_model}) должно делиться на n_heads_q ({self.n_heads_q})"
            )
        if self.n_heads_q % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads_q ({self.n_heads_q}) должно делиться на n_kv_heads ({self.n_kv_heads}) "
                "иначе GQA-grouping невозможен."
            )
        if self.n_active_experts > self.n_experts:
            raise ValueError("n_active_experts не может превышать n_experts")

    # ── Удобные свойства ────────────────────────────────────────────────────
    @property
    def head_dim_computed(self) -> int:
        """Размер одного attention head: d_model // n_heads_q (== 96 для 768/8)."""
        return self.d_model // self.n_heads_q

    @property
    def n_groups(self) -> int:
        """Число групп GQA: n_heads_q // n_kv_heads (по скольким q-head общий kv)."""
        return self.n_heads_q // self.n_kv_heads

    @property
    def ffn_hidden(self) -> int:
        """Внутренняя размерность SwiGLU-FFN одного эксперта."""
        # SwiGLU-стиль: расширяем ~2x c округлением вверх до кратного 8 (удобно для GEMM).
        hidden = int(self.d_model * self.ffn_expand)
        return (hidden + 7) // 8 * 8


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация токенизатора.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TokenizerConfig:
    """Настройки обучения токенизатора — ОФИЦИАЛЬНЫЙ SentencePiece BPE.

    Инженерный контекст (аудит Llama-3 / Qwen-3 / DeepSeek / Gemma-3 / Mistral-Tekken /
    SmolLM2 / Phi-4 / GPT-4o): индустрия ушла от SentencePiece к byte-level BPE
    (tiktoken/HF) прежде всего ради ① сохранения whitespace кода (переносы/табы/отступы)
    и ② врождённого byte-fallback (нет <unk>). ТЗ требует официальный SentencePiece —
    поэтому воспроизводим это поведение зрелыми флагами самого SentencePiece:
        • byte_fallback=True          — OOV-символ → его UTF-8 байты (<0xNN>); <unk> почти
                                        не эмиттится → поведение как у byte-level BPE (GPT-4).
        • normalization="identity"    — НЕ NFKC: nmt_nfkc в SP коллапсирует \\n/табы/отступы
                                        и ломает код. identity сохраняет байты точно.
        • add_dummy_prefix=False      — без ведущего ▁ (нет «leading-space quirk»; как SmolLM2/Mistral).
        • remove_extra_whitespaces=False — сохраняем кратные пробелы/отступы (критично для Python).
        • split_digits=True           — каждая цифра отдельным токеном (числа/математика/код).
        • split_by_unicode_script=True — границы токенов по смене Unicode-скрипта.
        • character_coverage=0.9995   — 99.95% символов в словарь, хвост → byte fallback.
    Поля ниже — реальные аргументы sentencepiece.SentencePieceTrainer.train().
    """

    # ── Размеры и тип модели ────────────────────────────────────────────────
    vocab_size: int = 65_536          # спецтокены + BPE-merge + 256 байт byte-fallback
    model_type: str = "bpe"           # именно BPE (как в ТЗ), не unigram/word/char
    character_coverage: float = 0.9995
    hard_vocab_limit: bool = True     # ровно vocab_size пиктограмм (256 байт byte-fallback в хвосте)

    # ── Инженерные флаги (обоснования — в докстринге выше и в tokenizer_train.py) ─
    byte_fallback: bool = True
    normalization: str = "identity"   # передаётся как normalization_rule_name
    add_dummy_prefix: bool = False
    remove_extra_whitespaces: bool = False
    allow_whitespace_only_pieces: bool = True   # пиктограммы из одних пробелов (отступы)
    split_digits: bool = True
    split_by_unicode_script: bool = True
    split_by_whitespace: bool = True

    # ── Ресурсы обучения ─────────────────────────────────────────────────────
    input_sentence_size: int = 0      # 0 — использовать все строки корпуса (без подвыборки)
    max_sentence_length: int = 32_768 # дефолт SP=4192 → длинные строки кода/SQL пропускаются; поднимаем
    num_threads: int = 4
    num_sub_iterations: int = 2
    shuffle_input: bool = True        # у SentencePiece нет seed → воспроизводимость даёт детерминированный combined.txt

    # ── Спецтокены: строки и ID синхронизированы с SpecialTokens ────────────
    pad_token: str = "<|pad|>"
    bos_token: str = "<|bos|>"
    eos_token: str = "<|eos|>"
    unk_token: str = "<|unk|>"
    pad_id: int = SpecialTokens.PAD   # 0
    bos_id: int = SpecialTokens.BOS   # 1
    eos_id: int = SpecialTokens.EOS   # 2
    unk_id: int = SpecialTokens.UNK   # 3
    add_bos: bool = False             # BOS/EOS добавляем вручную в dataset.py (контроль loss-маски)
    add_eos: bool = False

    # ── Пути: корпус и артефакты ────────────────────────────────────────────
    corpus_dir: str = "tokenizer_corpus"                     # откуда собирать тексты/код
    combined_corpus: str = "tokenizer_corpus/_combined.txt"  # объединённый корпус для обучения
    output_dir: str = "tokenizer"                            # → tokenizer/tokenizer.model + .vocab
    model_name: str = "tokenizer"                            # префикс артефактов


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация датасета и смешивания.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DatasetSource:
    """Один источник данных с HuggingFace."""
    name: str
    path: str                          # hf path, напр. "HuggingFaceFW/fineweb-edu"
    subset: Optional[str] = None       # config/subset (напр. "sample-10BT")
    split: str = "train"
    kind: str = "pretrain"             # "pretrain" (сырой текст → causal LM) | "instruct" (диалог с loss-маской)
    text_field: Optional[str] = None   # поле с текстом для kind="pretrain" (напр. "text"/"code")
    conversation_field: Optional[str] = None  # поле с диалогом для kind="instruct" (напр. "conversations")
    instruct_format: Optional[str] = None  # как парсить instruct: "sharegpt" | "magicoder" | "alpaca"
    language: str = "en"               # 'en'|'code' — для контроля пропорций (v1 без 'ru')
    weight: float = 0.0               # доля в финальной смеси (нормируется в dataset.py)


@dataclass
class DatasetConfig:
    """
    Смешивание данных v1 — только English + Code (без русского).

    Пропорции (рецепт в духе SmolLM2 — ближайший аналог малой EN+Code модели):
        50 % Code    = 40 % pretrain (github-code) + 10 % instruct (Magicoder)
        50 % English = 40 % pretrain (FineWeb-Edu)  + 10 % chat     (OpenHermes)
    Реальное смешивание идёт по per-source weight (нормируется в dataset.py);
    code_ratio/en_ratio ниже — высокоуровневая сверка намерения.
    """
    sources: List[DatasetSource] = field(default_factory=list)
    seed: int = 42

    # Высокоуровневые доли (сумма = 1.0). Фактическое смешивание — по weight источников.
    code_ratio: float = 0.50
    en_ratio: float = 0.50

    # ── Потоковый режим (Lightning.ai: не тянем терабайты на диск) ──────────
    streaming: bool = True
    tokenizer_model_path: str = "tokenizer/tokenizer.model"  # обученный SentencePiece
    hf_token_env: str = "HF_TOKEN"         # env-переменная с HuggingFace-токеном (для gated датасетов)

    max_total_tokens: int = 5_000_000_000  # потолок токенов на прогон
    shuffle_buffer: int = 10_000           # размер shuffle-буфера для streaming
    preprocessing_workers: int = 4
    packing: bool = True                   # упаковка документов в окна max_seq_len (без PAD-расхода)
    min_doc_chars: int = 8                 # выкидывать слишком короткие документы
    val_take: int = 2_000                  # сколько первых упакованных окон отложить под валидацию

    # Куда кэшировать обработанные данные на диске.
    cache_dir: str = "data_cache"

    def __post_init__(self) -> None:
        s = self.code_ratio + self.en_ratio
        if abs(s - 1.0) > 1e-3:
            raise ValueError(f"Сумма code_ratio+en_ratio должна быть 1.0, а получилась {s}")


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация обучения.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """Цикл обучения: оптимизация, память, чекпойнты, логирование."""

    # ── Оптимизатор и расписание ────────────────────────────────────────────
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # ── Расписание LR ────────────────────────────────────────────────────────
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_steps: int = 1_000
    min_lr_ratio: float = 0.1         # финальный LR = lr * min_lr_ratio
    total_steps: int = 50_000          # этаж сетки (реально бьётся по эпохам/токенам в train.py)

    # ── Батч и аккумуляция (по умолчанию = профиль «Рабочая лошадка», T4 15GB) ──
    micro_batch_size: int = 2          # батч в одну forward (T4 15GB)
    grad_accum_steps: int = 8          # аккумуляция → effective batch = 2*8 = 16
    effective_batch_size: int = 16     # пересчитывается в __post_init__

    # ── Точность, компиляция и память ─────────────────────────────────────────
    dtype: str = "fp16"                # "fp16"|"bf16"|"fp32"; T4→fp16+GradScaler, A100→bf16
    grad_checkpointing: bool = True    # экономия активаций ценой рекомпьюта (T4:True, A100:можно False)
    use_flash_attention: bool = True   # дублирует флаг из ModelConfig (для train.py / chat.py)
    dynamic_loss_scale: bool = True    # fp16: динамический GradScaler (bf16 не нужен)
    use_torch_compile: bool = False    # torch.compile(): на A100 «Ракета» ускоряет, на T4 обычно нет
    torch_compile_mode: str = "default"  # "default"|"reduce-overhead"|"max-autotune"

    # ── Платформа и профиль железа (Lightning.ai; см. HARDWARE_PROFILES ниже) ──
    platform: str = "lightning"          # целевая платформа: Lightning.ai Studios
    hardware_profile: str = "workhorse"  # "workhorse" (T4 15GB) | "rocket" (A100 40GB)

    # ── Воспроизводимость ───────────────────────────────────────────────────
    seed: int = 42

    # ── Чекпойнты и сохранение ──────────────────────────────────────────────
    output_dir: str = "checkpoints"
    save_every_steps: int = 500
    keep_last_n: int = 3               # сколько последних чекпойнтов держать на диске
    # Lightning.ai Studios: рабочая папка стьюдии уже персистентна между сессиями.
    # При желании дублировать чекпойнты в общий Teamspace-драйв — укажите путь, иначе None.
    backup_dir: Optional[str] = None   # напр. "/teamspace/studios/this_studio/lunami_ckpt"

    # ── Логирование ──────────────────────────────────────────────────────────
    log_interval: int = 20             # печатать лосс каждые N шагов
    eval_interval: int = 500           # полный eval на val-сплите
    eval_steps: int = 100              # число eval-батчей
    use_wandb: bool = True
    wandb_project: str = "lunami-mini"
    wandb_entity: Optional[str] = None

    # ── Возобновление ────────────────────────────────────────────────────────
    resume_from: Optional[str] = None   # путь к чекпойнту для догонки (None — с нуля)

    def __post_init__(self) -> None:
        # Гарантируем согласованность effective-батча.
        self.effective_batch_size = self.micro_batch_size * self.grad_accum_steps


# ─────────────────────────────────────────────────────────────────────────────
# Гибридная стратегия Lightning.ai: два профиля железа.
#   «Ракета»          (A100 40GB): bf16 + большой батч + torch.compile — максимум скорости.
#   «Рабочая лошадка» (T4 15GB)  : fp16 + GradScaler + grad-checkpointing — влезаем в память.
# train.py выбирает профиль (--profile rocket|workhorse) и применяет apply_hardware_profile().
# ─────────────────────────────────────────────────────────────────────────────
HARDWARE_PROFILES: Dict[str, Dict] = {
    "rocket": {          # A100 40GB — «Ракета»
        "dtype": "bf16",
        "micro_batch_size": 16,
        "grad_accum_steps": 2,          # effective batch = 32
        "use_torch_compile": True,
        "torch_compile_mode": "max-autotune",
        "dynamic_loss_scale": False,    # bf16 не требует GradScaler
        "grad_checkpointing": False,    # на A100 хватает памяти → быстрее без рекомпьюта
    },
    "workhorse": {       # T4 15GB — «Рабочая лошадка»
        "dtype": "fp16",
        "micro_batch_size": 2,
        "grad_accum_steps": 8,          # effective batch = 16
        "use_torch_compile": False,     # на T4 компиляция долгая и мало что даёт
        "torch_compile_mode": "default",
        "dynamic_loss_scale": True,     # fp16 → нужен динамический GradScaler
        "grad_checkpointing": True,     # экономим память ценой рекомпьюта
    },
}


def apply_hardware_profile(train: TrainConfig, profile: str) -> TrainConfig:
    """
    Применяет профиль железа ('rocket' | 'workhorse') к TrainConfig НА МЕСТЕ и возвращает его.
    Пересчитывает effective_batch_size. Используется в train.py при старте.
    """
    if profile not in HARDWARE_PROFILES:
        raise ValueError(
            f"Неизвестный профиль '{profile}'. Доступны: {list(HARDWARE_PROFILES)}"
        )
    for key, value in HARDWARE_PROFILES[profile].items():
        setattr(train, key, value)
    train.hardware_profile = profile
    train.__post_init__()  # пересчитать effective_batch_size под новый micro/accum
    return train


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация инференса / чата.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InferenceConfig:
    """Параметры генерации и формат чата."""
    checkpoint_path: str = "checkpoints/latest"
    tokenizer_path: str = "tokenizer"
    device: str = "cuda"               # "cuda"|"cpu"; на T4 — cuda
    dtype: str = "fp16"

    # ── Декодирование ────────────────────────────────────────────────────────
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0                     # 0 — без top-k ограничения
    do_sample: bool = True
    repetition_penalty: float = 1.15
    eos_token_id: int = SpecialTokens.EOS
    pad_token_id: int = SpecialTokens.PAD

    # ── Формат чата (строки специальных токенов уже есть в SpecialTokens) ────
    system_prompt: str = (
        "Ты — Лунами, умный дружелюбный ассистент и опытный программист. "
        "Отвечай ясно и по делу. Если задача про код — давай работающий код, "
        "объясняй кратко на русском или английском по языку собеседника."
    )
    max_context_turns: int = 8         # сколько прошлых реплик учитывать в истории


# ─────────────────────────────────────────────────────────────────────────────
# Корневой конфиг (агрегатор) — удобен для передачи в один dataclass-объект.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LunamiConfig:
    """Вся конфигурация в одном месте."""
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    version: str = "1.0.0"
    arch: str = "hybrid-mamba2-gqa-moe"

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Сохраняем конфиг в JSON (полезно рядом с чекпойнтом)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LunamiConfig":
        """Загрузка конфига из JSON. Используется в chat.py / train.py при resume."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
            tokenizer=TokenizerConfig(**data.get("tokenizer", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            train=TrainConfig(**data.get("train", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            version=data.get("version", "1.0.0"),
            arch=data.get("arch", "hybrid-mamba2-gqa-moe"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции: сидирование, оценка параметров, список источников.
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Фиксируем все ГПСЧ для воспроизводимости: python, numpy, (torch — в train.py)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def default_dataset_sources() -> List[DatasetSource]:
    """
    Лучшие бесплатные HuggingFace-датасеты для EN+Code (v1). Рецепт — как у SmolLM2.

    Итог: 50 % Code / 50 % English.
      • English pretrain — FineWeb-Edu: образовательный веб, топ-качество для маленьких моделей.
      • Code pretrain    — codeparrot-clean-train (ungated, Python). Апгрейд/мультиязычность:
                           bigcode/starcoderdata (gated — нужен HF-токен и принятие лицензии).
      • Code instruct    — Magicoder-OSS-Instruct-75K: учит следовать инструкциям по коду.
      • English chat     — OpenHermes-2.5: учит формат чата <|user|>/<|assistant|>.

    ПРИМЕЧАНИЕ: живой аудит (2026-07-07) показал, что codeparrot/github-code-clean больше НЕ
    грузится современной `datasets` (использует устаревший loading-script формат: "Dataset
    scripts are no longer supported"). Заменено на codeparrot/codeparrot-clean-train — публичный,
    проверен вживую (streaming, поле "content"). Минус: только Python, а не мультиязычный код —
    приемлемо для v1 (Python — самый частый язык), апгрейд на мультиязычность см. выше.
    Прочие subset/поля датасетов на HuggingFace тоже могут меняться — dataset.py грузит их
    защищённо и при несовпадении даёт понятную ошибку. Проверьте доступ (и HF_TOKEN для gated).
    """
    return [
        # ── English pretrain (40%) ──────────────────────────────────────────
        DatasetSource(
            name="fineweb_edu",
            path="HuggingFaceFW/fineweb-edu",
            subset="sample-10BT",      # ~10B токенов сэмпл; для полного — убрать subset
            split="train",
            kind="pretrain",
            text_field="text",
            language="en",
            weight=0.40,
        ),
        # ── Code pretrain (40%) ─────────────────────────────────────────────
        DatasetSource(
            name="codeparrot_clean",
            path="codeparrot/codeparrot-clean-train",
            subset=None,
            split="train",
            kind="pretrain",
            text_field="content",
            language="code",
            weight=0.40,
        ),
        # ── Code instruct (10%) ─────────────────────────────────────────────
        DatasetSource(
            name="magicoder_oss",
            path="ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            kind="instruct",
            instruct_format="magicoder",   # поля problem/solution
            language="code",
            weight=0.10,
        ),
        # ── English chat (10%) ──────────────────────────────────────────────
        DatasetSource(
            name="openhermes_2_5",
            path="teknium/OpenHermes-2.5",
            split="train",
            kind="instruct",
            conversation_field="conversations",
            instruct_format="sharegpt",    # список {from, value}
            language="en",
            weight=0.10,
        ),
    ]


def count_parameters(cfg: ModelConfig) -> Dict[str, int]:
    """
    Оценка числа параметров модели по конфигуляции.

    Возвращает словарь с разбивкой и два итоговых числа:
        'active'  — параметры, реально «работающие» на токен (2 из 8 экспертов MoE);
        'total'   — все хранящиеся параметры (все 8 экспертов MoE).

    NB: оценка грубая (без учёта малых весов нормализации, bias, conv-фильтров),
    но даёт порядок величины для сверки с ТЗ («~125M активных»).
    """
    d = cfg.d_model
    V = cfg.vocab_size

    # 1) Эмбеддинги. При tie_embeddings lm_head == embedding → считается один раз.
    if cfg.tie_embeddings:
        embed = V * d
    else:
        embed = 2 * V * d

    # 2) Mamba-2 блок на слой:
    #    in_proj  : d -> 2*d_inner   (x + z),  d_inner = expand*d
    #    conv1d   : d_inner * k
    #    out_proj : d_inner -> d
    #    dt/A/D   : ~d_inner (пренебрежимо мало)
    d_inner = cfg.mamba_expand * d
    mamba_per_layer = (2 * d * d_inner) + (d_inner * cfg.mamba_conv_kernel) + (d_inner * d)
    mamba_total = mamba_per_layer * cfg.n_mamba

    # 3) Attention (GQA) на слой:
    #    Q: d -> n_q*head_dim = d,   K/V: d -> n_kv*head_dim,   O: d -> d
    head_dim = d // cfg.n_heads_q
    kv_dim = cfg.n_kv_heads * head_dim
    attn_per_layer = (d * d) + 2 * (d * kv_dim) + (d * d)
    attn_total = attn_per_layer * cfg.n_attention

    # 4) FFN/MoE на слой. SwiGLU: gate(d->h) + up(d->h) + down(h->d) = 3*d*h на эксперт.
    h = cfg.ffn_hidden
    ffn_per_expert = 3 * d * h
    if cfg.use_moe:
        # Плотный роутер: d -> n_experts (один на слой).
        router = d * cfg.n_experts
        ffn_total_all = (ffn_per_expert * cfg.n_experts + router) * cfg.n_attention
        ffn_active = (ffn_per_expert * cfg.n_active_experts + router) * cfg.n_attention
    else:
        ffn_total_all = ffn_per_expert * cfg.n_attention
        ffn_active = ffn_total_all

    total = embed + mamba_total + attn_total + ffn_total_all
    active = embed + mamba_total + attn_total + ffn_active

    return {
        "embeddings": embed,
        "mamba_total": mamba_total,
        "attention_total": attn_total,
        "ffn_total_all": ffn_total_all,
        "ffn_active": ffn_active,
        "TOTAL_all": total,
        "ACTIVE_per_token": active,
    }


def human_readable(n: int) -> str:
    """Преобразует число в человекочитаемый вид, напр. 125_000_000 → '125.00M'."""
    for unit, divisor in (("B", 1), ("K", 1e3), ("M", 1e6), ("G", 1e9)):
        if n < divisor * 1000:
            return f"{n / divisor:.2f}{unit}"
    return f"{n / 1e9:.2f}G"


# ─────────────────────────────────────────────────────────────────────────────
# Готовая дефолтная конфигурация и точка входа для быстрого осмотра.
# ─────────────────────────────────────────────────────────────────────────────
def get_default_config() -> LunamiConfig:
    """Возвращает полностью настроенный LunamiConfig со всеми источниками данных."""
    cfg = LunamiConfig()
    cfg.dataset.sources = default_dataset_sources()
    return cfg


if __name__ == "__main__":
    # Быстрый sanity-check при запуске: печатаем оценку параметров.
    set_seed(42)
    model_cfg = ModelConfig()
    print("=" * 60)
    print("Lunami-Mini — sanity-check конфигурации")
    print("=" * 60)
    print(f"d_model              : {model_cfg.d_model}")
    print(f"Слоёв (Mamba/Attn)   : {model_cfg.n_layers} ({model_cfg.n_mamba}/{model_cfg.n_attention})")
    print(f"head_dim / group GQA : {model_cfg.head_dim_computed} / {model_cfg.n_groups}")
    print(f"FFN hidden (эксперт) : {model_cfg.ffn_hidden}")
    print()
    params = count_parameters(model_cfg)
    for k, v in params.items():
        print(f"{k:22s}: {v:>14,}  ({human_readable(v)})")
    print()
    print(f"Активных параметров на токен ≈ {human_readable(params['ACTIVE_per_token'])}")
    print(f"Всего хранящихся       ≈ {human_readable(params['TOTAL_all'])}")
    print("=" * 60)
