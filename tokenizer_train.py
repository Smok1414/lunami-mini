# -*- coding: utf-8 -*-
"""
tokenizer_train.py — обучение токенизатора «Lunami-Mini» (ОФИЦИАЛЬНЫЙ SentencePiece BPE).

Первая версия модели — максимально сильная на English + Code (без русского и прочих языков):
лучше одна очень сильная EN+Code модель, чем много посредственных языков.

═══════════════════════════════════════════════════════════════════════════════════════
ИНЖЕНЕРНЫЙ АУДИТ СОВРЕМЕННЫХ ТОКЕНИЗАТОРОВ (почему выбрана именно эта конфигурация)
═══════════════════════════════════════════════════════════════════════════════════════

Сравнение реально существующих зрелых решений (не эксперименты):

  ┌────────────────┬───────────────────────────┬────────┬─────────────────────────────────┐
  │ Модель         │ Движок                    │ Vocab  │ Ключевое инженерное решение     │
  ├────────────────┼───────────────────────────┼────────┼─────────────────────────────────┤
  │ Llama 3        │ tiktoken byte-level BPE    │ 128k   │ Ушли с SentencePiece (Llama1/2, │
  │                │                           │        │ 32k) ради кода/whitespace;      │
  │                │                           │        │ split-digits, без dummy-prefix  │
  │ Qwen 2.5/3     │ tiktoken byte-level BPE    │ ~151k  │ CJK по 1 токену; split-digits   │
  │ DeepSeek V3    │ byte-level BPE            │ ~100-128k│ оптимизация под код/математику  │
  │ Gemma 2/3      │ SentencePiece + byte_fb    │ 256k   │ byte_fallback + огромный словарь│
  │ Mistral        │ SP(v1) → Tekken(tiktoken) │ 32k→131k│ осознанно без dummy-prefix     │
  │ SmolLM2        │ HF byte-level BPE         │ ~49k   │ БЛИЖАЙШИЙ АНАЛОГ Lunami: малая  │
  │                │                           │        │ EN+Code модель, byte-level ради │
  │                │                           │        │ кода, без dummy-prefix          │
  │ Phi-3/4        │ tiktoken (cl100k)         │ ~100-320k│ byte-level                     │
  │ GPT-4 / 4o     │ tiktoken cl100k / o200k   │ 100k/200k│ регекс со split-digits, byte-lvl│
  └────────────────┴───────────────────────────┴────────┴─────────────────────────────────┘

ГЛАВНЫЙ ВЫВОД: вся индустрия ушла на byte-level BPE (tiktoken/HF) по двум причинам —
  ① сохранение whitespace кода (переносы строк, табы, отступы — критично для Python и т.п.);
  ② врождённый byte-fallback: любой символ кодируется в UTF-8 байты, <unk> не нужен.
ТЗ требует ОФИЦИАЛЬНЫЙ SentencePiece — поэтому мы воспроизводим это же поведение зрелыми
флагами самого SentencePiece. Каждое решение ниже — из реально работающих продакшн-решений.

───────────────────────────────────────────────────────────────────────────────────────
РЕШЕНИЕ 1. Byte Fallback = True
    Почему лучше : OOV-символ (редкий юникод, emoji, необычный код-символ) кодируется как
                   последовательность его UTF-8 байтов (<0xNN>) вместо <unk>. Именно так
                   byte-level BPE (GPT-4/Llama-3) не теряет ни одного символа.
    Качество     : ↑↑  — ноль потерь информации на редких символах, emoji, любом юникоде.
    Скорость     : ≈   — на обычном тексте не влияет; редкие символы дают чуть больше токенов.
    Память       : +256 байт-токенов в словаре (незначительно от 65536).
    Проверено    : round-trip 🚀/café/юникода на этом же коде — decode(encode(x)) == x.

РЕШЕНИЕ 2. Normalization = "identity"  (НЕ NFKC!)
    Почему лучше : NFKC (в SentencePiece — nmt_nfkc) вместе с полезной нормализацией
                   КОЛЛАПСИРУЕТ пробелы/переносы/табы → УБИВАЕТ отступы кода. Для EN+Code
                   это неприемлемо. identity сохраняет байты точно (переносы, табы, регистр).
    Качество кода: ↑↑  — Python-отступы, табы, переносы строк сохраняются побайтово.
    Компромисс   : нет нормализации лигатур/fullwidth — но для EN+ASCII-кода это неактуально.
    Примечание   : SmolLM2/Qwen используют NFKC через HF/tiktoken, где NFKC НЕ трогает
                   whitespace. В SentencePiece это связано → поэтому здесь identity.

РЕШЕНИЕ 3. Add Dummy Prefix = False
    Почему лучше : Llama-1/2 вставляли ведущий ▁ в начало текста → «leading-space quirk»
                   ("Hello" ≠ " Hello"). Llama-3/SmolLM2/Mistral это убрали. Границы реплик
                   задаём спецтокенами (<|user|> и т.п.), фантомный пробел не нужен.
    Качество     : ↑   — токенизация обратима и предсказуема, нет привязки к ведущему пробелу.

РЕШЕНИЕ 4. Remove Extra Whitespaces = False  +  Allow Whitespace-Only Pieces = True
    Почему лучше : по умолчанию SentencePiece СХЛОПЫВАЕТ кратные пробелы → уничтожает
                   Python-отступы. Ставим False. allow_whitespace_only_pieces позволяет
                   выучить пиктограммы из одних пробелов (напр. 4-пробельный отступ) —
                   большой выигрыш сжатия на коде.
    Качество кода: ↑↑  — отступы сохранены и эффективно кодируются.

РЕШЕНИЕ 5. Split Digits = True
    Почему лучше : каждая цифра 0-9 — отдельный базовый токен (Llama-3/Qwen/GPT-4/SmolLM2).
                   Числа/математика/литералы в коде обобщаются лучше (нет токена «1234»).
    Качество     : ↑↑ на числах/математике.  Скорость: длинные числа → чуть больше токенов.

РЕШЕНИЕ 6. Split by Unicode Script = True
    Почему лучше : граница токена при смене Unicode-скрипта. Для EN+Code (латиница + символы)
                   защищает от мусорных merge между разными системами письма.

РЕШЕНИЕ 7. Character Coverage = 0.9995
    Почему лучше : 99.95% символов корпуса попадают в словарь как базовые; хвост 0.05%
                   (экзотика) уходит в byte_fallback. EN+Code почти весь ASCII → покрытие полное.

РЕШЕНИЕ 8. Спецтокены BOS / EOS / PAD / UNK (+ chat/code)
    Почему так   : pad/bos/eos/unk резервируются сплошным блоком id 0..3 (требование SP);
                   структурные chat/code токены — через user_defined_symbols (id 4..9,
                   строго по порядку — эмпирически проверено). add_bos/add_eos=False:
                   BOS/EOS добавляем вручную в dataset.py для точного контроля loss-маски.

Раскладка (синхронизирована с config.SpecialTokens):
    0 <|pad|>  1 <|bos|>  2 <|eos|>  3 <|unk|>
    4 <|system|>  5 <|user|>  6 <|assistant|>  7 <|end|>  8 <|code|>  9 <|endofcode|>

═══════════════════════════════════════════════════════════════════════════════════════

Запуск:
    1) положить EN-текст и код в папку ./tokenizer_corpus/ (любая вложенность);
    2) python tokenizer_train.py
    3) на выходе: ./tokenizer/tokenizer.model  и  ./tokenizer/tokenizer.vocab
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── tqdm (прогресс-бары); мягкий фолбэк, чтобы скрипт не падал без пакета ──────────────
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []

# ── sentencepiece — официальная библиотека (ТЗ: НЕ реализуем BPE вручную) ──────────────
try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "ERROR: не установлен sentencepiece.\n"
        "Установите:  pip install sentencepiece tqdm\n"
    )
    sys.exit(1)

from config import SpecialTokens, TokenizerConfig, set_seed


# ═════════════════════════════════════════════════════════════════════════════════════
# Логирование
# ═════════════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tokenizer_train")


# ═════════════════════════════════════════════════════════════════════════════════════
# Расширения файлов корпуса: English-текст + исходный код (v1 — без прочих языков).
# ═════════════════════════════════════════════════════════════════════════════════════
TEXT_EXTENSIONS: frozenset = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".text",
})
CODE_EXTENSIONS: frozenset = frozenset({
    ".py", ".pyi", ".pyx",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".kt", ".kts", ".scala", ".groovy",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift", ".m", ".mm",
    ".sh", ".bash", ".zsh", ".ps1", ".bat",
    ".sql", ".r", ".lua", ".pl", ".pm", ".dart", ".ex", ".exs",
    ".clj", ".cljs", ".hs", ".ml", ".mli", ".erl", ".vue", ".svelte",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".xml", ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".make", ".mk", ".cmake", ".gradle", ".dockerfile",
})
ALLOWED_EXTENSIONS: frozenset = TEXT_EXTENSIONS | CODE_EXTENSIONS

# Файлы крупнее — обрабатываем, но предупреждаем (могут быть дампы/бинарь по ошибке).
LARGE_FILE_WARN_BYTES: int = 100 * 1024 * 1024  # 100 MB


# ═════════════════════════════════════════════════════════════════════════════════════
# Статистика корпуса.
# ═════════════════════════════════════════════════════════════════════════════════════
@dataclass
class CorpusStats:
    """Счётчики по сборке корпуса и по обученному токенизатору."""
    files_found: int = 0          # найдено файлов с допустимыми расширениями
    files_broken: int = 0         # пропущено (не читается / не UTF-8)
    files_duplicate: int = 0      # пропущено как точный дубликат (по sha256 содержимого)
    documents: int = 0            # реально использовано документов
    characters: int = 0           # всего символов записано в корпус
    lines: int = 0                # всего непустых строк записано
    bytes_written: int = 0        # всего UTF-8 байт записано
    # заполняется после обучения:
    corpus_tokens: int = 0        # всего токенов при кодировании корпуса
    bytes_per_token: float = 0.0  # метрика сжатия (чем больше, тем лучше)

    def log_summary(self) -> None:
        logger.info("─" * 70)
        logger.info("СТАТИСТИКА КОРПУСА")
        logger.info("  файлов найдено         : %s", f"{self.files_found:,}")
        logger.info("  файлов битых/не-UTF8   : %s", f"{self.files_broken:,}")
        logger.info("  файлов-дубликатов      : %s", f"{self.files_duplicate:,}")
        logger.info("  документов использовано: %s", f"{self.documents:,}")
        logger.info("  строк (непустых)       : %s", f"{self.lines:,}")
        logger.info("  символов               : %s", f"{self.characters:,}")
        logger.info("  байт (UTF-8)           : %s", f"{self.bytes_written:,}")
        logger.info("─" * 70)


# ═════════════════════════════════════════════════════════════════════════════════════
# 1. Поиск файлов корпуса.
# ═════════════════════════════════════════════════════════════════════════════════════
def discover_files(corpus_dir: str) -> List[Path]:
    """
    Рекурсивно находит все файлы с допустимыми расширениями в corpus_dir.

    Возвращает ДЕТЕРМИНИРОВАННО ОТСОРТИРОВАННЫЙ список путей — это важно для
    воспроизводимости: у SentencePiece нет параметра seed, поэтому единственный способ
    получить стабильный результат — подать байт-в-байт одинаковый combined.txt.
    """
    root = Path(corpus_dir)
    if not root.exists():
        return []
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in ALLOWED_EXTENSIONS:
                files.append(p)
    files.sort(key=lambda p: str(p).lower())  # детерминированный порядок
    return files


# ═════════════════════════════════════════════════════════════════════════════════════
# 2. Очистка + объединение в единый корпус.
# ═════════════════════════════════════════════════════════════════════════════════════
def build_combined_corpus(files: List[Path], out_path: str) -> CorpusStats:
    """
    Объединяет файлы в один combined.txt (одна строка = один обучающий пример для SP).

    Делает:
      • пропускает битые/не-UTF-8 файлы (ловим UnicodeDecodeError / OSError);
      • дедуп на уровне ДОКУМЕНТОВ по sha256 содержимого (память O(#уникальных файлов));
        НЕ построчно — построчный дедуп убил бы частоты '}', 'return', … и навредил бы коду;
      • выкидывает пустые/пробельные строки (для обучения словаря они бесполезны);
      • отступы ВНУТРИ строк сохраняются (identity + remove_extra_whitespaces=False),
        поэтому пиктограммы отступов остаются обучаемыми.

    Замечание про переносы строк: SentencePiece читает корпус построчно, поэтому '\\n'
    не попадает внутрь обучающих примеров (это разделитель). Но благодаря byte_fallback
    '\\n' (<0x0A>) и '\\t' (<0x09>) ВСЕГДА кодируемы и обратимы на инференсе — проверено
    в verify_tokenizer(). Это осознанный компромисс мандата «официальный SentencePiece».
    """
    stats = CorpusStats(files_found=len(files))
    seen_hashes: set[str] = set()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8", newline="\n") as w:
        for path in tqdm(files, desc="Сборка корпуса", unit="файл"):
            # 1) читаем сырые байты, ловим ошибки чтения
            try:
                raw = path.read_bytes()
            except OSError as e:
                logger.warning("Не удалось прочитать %s: %s", path, e)
                stats.files_broken += 1
                continue

            if len(raw) > LARGE_FILE_WARN_BYTES:
                logger.warning("Крупный файл %.1f MB: %s", len(raw) / 1e6, path)

            # 2) дедуп по содержимому (sha256 сырых байт — дёшево по памяти)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen_hashes:
                stats.files_duplicate += 1
                continue
            seen_hashes.add(digest)

            # 3) декодируем строго как UTF-8; не декодируется → считаем битым/бинарным
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                stats.files_broken += 1
                continue

            # 4) построчная запись без пустых/пробельных строк
            doc_had_content = False
            for line in text.splitlines():
                if not line.strip():
                    continue  # пустая или пробельная строка — пропуск
                w.write(line)
                w.write("\n")
                stats.lines += 1
                stats.characters += len(line)
                stats.bytes_written += len(line.encode("utf-8")) + 1  # +1 за '\n'
                doc_had_content = True

            if doc_had_content:
                stats.documents += 1

    return stats


# ═════════════════════════════════════════════════════════════════════════════════════
# 3. Сбор аргументов SentencePiece Trainer из TokenizerConfig.
# ═════════════════════════════════════════════════════════════════════════════════════
def build_trainer_kwargs(cfg: TokenizerConfig, input_path: str, model_prefix: str) -> Dict:
    """
    Отображает поля TokenizerConfig на реальные аргументы SentencePieceTrainer.train().

    user_defined_symbols — это структурные chat/code токены (id 4..9): берём из
    SpecialTokens.names() всё, что НЕ входит в резерв pad/bos/eos/unk, по возрастанию id.
    Порядок критичен: SentencePiece назначает им id строго по порядку списка (проверено).
    """
    reserved = {SpecialTokens.PAD, SpecialTokens.BOS, SpecialTokens.EOS, SpecialTokens.UNK}
    user_defined_symbols: List[str] = [
        name for tid, name in sorted(SpecialTokens.names().items()) if tid not in reserved
    ]

    return dict(
        input=input_path,
        model_prefix=model_prefix,
        # размеры / тип
        vocab_size=cfg.vocab_size,
        model_type=cfg.model_type,
        character_coverage=cfg.character_coverage,
        hard_vocab_limit=cfg.hard_vocab_limit,
        # инженерные флаги (см. аудит в докстринге модуля)
        byte_fallback=cfg.byte_fallback,
        normalization_rule_name=cfg.normalization,     # "identity"
        add_dummy_prefix=cfg.add_dummy_prefix,
        remove_extra_whitespaces=cfg.remove_extra_whitespaces,
        allow_whitespace_only_pieces=cfg.allow_whitespace_only_pieces,
        split_digits=cfg.split_digits,
        split_by_unicode_script=cfg.split_by_unicode_script,
        split_by_whitespace=cfg.split_by_whitespace,
        # ресурсы
        input_sentence_size=cfg.input_sentence_size,
        max_sentence_length=cfg.max_sentence_length,
        num_threads=cfg.num_threads,
        num_sub_iterations=cfg.num_sub_iterations,
        shuffle_input_sentence=cfg.shuffle_input,
        # спецтокены: резерв pad/bos/eos/unk (id 0..3)
        pad_id=cfg.pad_id, pad_piece=cfg.pad_token,
        bos_id=cfg.bos_id, bos_piece=cfg.bos_token,
        eos_id=cfg.eos_id, eos_piece=cfg.eos_token,
        unk_id=cfg.unk_id, unk_piece=cfg.unk_token,
        # структурные chat/code токены (id 4..9)
        user_defined_symbols=user_defined_symbols,
    )


# ═════════════════════════════════════════════════════════════════════════════════════
# 4. Обучение SentencePiece.
# ═════════════════════════════════════════════════════════════════════════════════════
def train_sentencepiece(cfg: TokenizerConfig) -> str:
    """Обучает SentencePiece BPE и сохраняет tokenizer.model + tokenizer.vocab."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(out_dir / cfg.model_name)

    kwargs = build_trainer_kwargs(cfg, cfg.combined_corpus, model_prefix)

    logger.info("Запуск обучения SentencePiece (vocab=%s, model_type=%s)...",
                f"{cfg.vocab_size:,}", cfg.model_type)
    logger.info("  user_defined_symbols = %s", kwargs["user_defined_symbols"])
    try:
        spm.SentencePieceTrainer.train(**kwargs)
    except RuntimeError as e:
        # Самая частая ошибка на бедном корпусе при hard_vocab_limit=True:
        # «Vocabulary size too high (N). Please set it to a value <= M».
        # hard_vocab_limit нам НУЖЕН (модели требуется ровно vocab_size),
        # поэтому не понижаем словарь молча — даём понятную инструкцию.
        msg = str(e)
        if "Vocabulary size too high" in msg:
            raise RuntimeError(
                f"Корпус слишком мал/беден для словаря {cfg.vocab_size:,}.\n"
                f"    SentencePiece: {msg.strip().splitlines()[-1]}\n"
                f"    Решение: добавьте больше English-текста и кода в '{cfg.corpus_dir}' "
                f"(для 65536 нужны сотни МБ+ разнообразного EN+Code),\n"
                f"    либо временно уменьшите TokenizerConfig.vocab_size."
            ) from e
        raise

    model_path = f"{model_prefix}.model"
    logger.info("Готово. Артефакты: %s(.model/.vocab)", model_prefix)
    return model_path


# ═════════════════════════════════════════════════════════════════════════════════════
# 5. Верификация (самоаудит в коде).
# ═════════════════════════════════════════════════════════════════════════════════════
# Наборы для round-trip: покрывают код/табы/переносы/emoji/юникод/UTF-8/длинный документ.
_ROUNDTRIP_CASES: Dict[str, str] = {
    "code_tabs_newlines": "def f(x):\n\tif x > 0:\n\t\treturn x\n\treturn -x\n",
    "code_spaces_indent": "class A:\n    def m(self):\n        return 1234567890\n",
    "json":               '{"key": "value", "nums": [1, 2, 3], "nested": {"a": true}}',
    "emoji":              "deploy 🚀 build passed ✅ fire 🔥",
    "unicode_accents":    "café naïve résumé — “smart quotes” and ellipsis…",
    "mixed_whitespace":   "a\t\tb    c\n\n\nd  ",
    "utf8_bytes":         "Ω≈ç√∫˜µ≤≥÷ ← стрелки и математика",
    "long_line":          ("x = " + "a + " * 500 + "z"),
}


def verify_tokenizer(model_path: str, cfg: TokenizerConfig) -> Tuple[bool, List[str]]:
    """
    Полный самоаудит обученного токенизатора. Возвращает (ok, список_проблем).

    Проверяет:
      • корректную загрузку модели;
      • размер словаря == vocab_size;
      • ID и строки ВСЕХ спецтокенов == config.SpecialTokens (fail-loud с диагностикой);
      • pad/bos/eos/unk id() совпадают;
      • round-trip decode(encode(x)) == x на коде/emoji/юникоде/пробелах/длинном тексте.
    """
    problems: List[str] = []

    # — загрузка —
    try:
        sp = spm.SentencePieceProcessor(model_file=model_path)
    except Exception as e:  # noqa: BLE001
        return False, [f"не удалось загрузить {model_path}: {e}"]

    # — размер словаря —
    vsize = sp.get_piece_size()
    if vsize != cfg.vocab_size:
        problems.append(f"vocab_size: ожидалось {cfg.vocab_size}, получено {vsize}")

    # — спецтокены: строка → id —
    for tid, name in sorted(SpecialTokens.names().items()):
        got = sp.piece_to_id(name)
        if got != tid:
            problems.append(f"спецтокен {name!r}: ожидался id={tid}, получен id={got}")
        # обратная проверка: id → строка
        back = sp.id_to_piece(tid)
        if back != name:
            problems.append(f"id {tid}: ожидалась строка {name!r}, получена {back!r}")

    # — встроенные id —
    for label, got, exp in (
        ("pad_id", sp.pad_id(), SpecialTokens.PAD),
        ("bos_id", sp.bos_id(), SpecialTokens.BOS),
        ("eos_id", sp.eos_id(), SpecialTokens.EOS),
        ("unk_id", sp.unk_id(), SpecialTokens.UNK),
    ):
        if got != exp:
            problems.append(f"{label}: ожидалось {exp}, получено {got}")

    # — round-trip —
    for name, text in _ROUNDTRIP_CASES.items():
        ids = sp.encode(text, out_type=int)
        back = sp.decode(ids)
        if back != text:
            problems.append(
                f"round-trip [{name}] не совпал:\n"
                f"       orig={text!r}\n"
                f"       back={back!r}"
            )

    return (len(problems) == 0), problems


# ═════════════════════════════════════════════════════════════════════════════════════
# 6. Метрика сжатия + примеры.
# ═════════════════════════════════════════════════════════════════════════════════════
def measure_corpus_tokens(model_path: str, corpus_path: str, stats: CorpusStats) -> None:
    """Кодирует корпус, считает суммарное число токенов и bytes/token (метрика сжатия)."""
    sp = spm.SentencePieceProcessor(model_file=model_path)
    total_tokens = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Подсчёт токенов корпуса", unit="стр"):
            line = line.rstrip("\n")
            if line:
                total_tokens += len(sp.encode(line, out_type=int))
    stats.corpus_tokens = total_tokens
    stats.bytes_per_token = (stats.bytes_written / total_tokens) if total_tokens else 0.0


def show_examples(model_path: str) -> None:
    """Демонстрация encode/decode и таблицы спецтокенов."""
    sp = spm.SentencePieceProcessor(model_file=model_path)

    logger.info("─" * 70)
    logger.info("СПЕЦТОКЕНЫ (id → строка → проверка piece_to_id)")
    for tid, name in sorted(SpecialTokens.names().items()):
        logger.info("  %2d  %-14s  piece_to_id=%d", tid, name, sp.piece_to_id(name))

    sample = "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n"
    ids = sp.encode(sample, out_type=int)
    pieces = [sp.id_to_piece(i) for i in ids]
    logger.info("─" * 70)
    logger.info("ПРИМЕР ТОКЕНИЗАЦИИ (код)")
    logger.info("  исходник : %r", sample)
    logger.info("  токенов  : %d", len(ids))
    logger.info("  pieces   : %s", pieces)
    logger.info("  ids      : %s", ids)
    logger.info("─" * 70)
    logger.info("ПРИМЕР ДЕТОКЕНИЗАЦИИ")
    logger.info("  decode(ids) : %r", sp.decode(ids))
    logger.info("  совпадает   : %s", sp.decode(ids) == sample)
    logger.info("─" * 70)


# ═════════════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════════════
def main() -> int:
    # На Windows консоль часто cp1251 → emoji/юникод/▁ в print падают. Переключаем UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    cfg = TokenizerConfig()
    set_seed(42)  # воспроизводимость (питон/нумпай); combined.txt детерминирован сортировкой

    logger.info("═" * 70)
    logger.info("ОБУЧЕНИЕ ТОКЕНИЗАТОРА LUNAMI-MINI (SentencePiece BPE, EN+Code)")
    logger.info("═" * 70)

    # 1) поиск файлов
    files = discover_files(cfg.corpus_dir)
    if not files:
        logger.error("В папке %r нет подходящих файлов.", cfg.corpus_dir)
        logger.error("Положите English-текст и/или код (%s …) в %r и повторите.",
                     ", ".join(sorted(list(ALLOWED_EXTENSIONS))[:8]), cfg.corpus_dir)
        return 1
    logger.info("Найдено файлов: %s", f"{len(files):,}")

    # 2) сборка корпуса
    stats = build_combined_corpus(files, cfg.combined_corpus)
    if stats.documents == 0 or stats.lines == 0:
        logger.error("После очистки корпус пуст (документов=%d, строк=%d). "
                     "Проверьте содержимое %r.", stats.documents, stats.lines, cfg.corpus_dir)
        return 1

    # 3) обучение
    model_path = train_sentencepiece(cfg)

    # 4) верификация (самоаудит)
    ok, problems = verify_tokenizer(model_path, cfg)

    # 5) метрика + примеры
    measure_corpus_tokens(model_path, cfg.combined_corpus, stats)
    stats.log_summary()
    logger.info("  токенов в корпусе      : %s", f"{stats.corpus_tokens:,}")
    logger.info("  сжатие (bytes/token)   : %.3f", stats.bytes_per_token)
    logger.info("  размер словаря         : %s", f"{cfg.vocab_size:,}")
    show_examples(model_path)

    # 6) итог верификации
    if ok:
        logger.info("✅ ВЕРИФИКАЦИЯ ПРОЙДЕНА: спецтокены, размеры и round-trip — OK.")
        logger.info("Артефакты: %s.model + %s.vocab",
                    str(Path(cfg.output_dir) / cfg.model_name),
                    str(Path(cfg.output_dir) / cfg.model_name))
        return 0

    logger.error("❌ ВЕРИФИКАЦИЯ НЕ ПРОЙДЕНА — найдены проблемы:")
    for p in problems:
        logger.error("  • %s", p)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
