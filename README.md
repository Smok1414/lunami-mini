# Lunami-Mini

Независимая LLM с нуля — не файнтюн и не обёртка над существующей моделью. Одна сфокусированная **English + Code** модель: гибрид Mamba-2 + GQA-Transformer + Mixture-of-Experts, ~165-170M активных параметров.

## Архитектура

- 24 слоя: `[Mamba, Mamba, Mamba, Attention] × 6` (75% Mamba-2 / 25% GQA-внимание)
- `d_model=768`, контекст 8192 токенов
- **Mamba-2**: state dim 64, expand 2, conv kernel 4 — chunkwise selective scan на чистом PyTorch (без зависимости от `mamba_ssm`)
- **GQA-внимание**: 8 query heads / 2 KV heads, head_dim 96, RoPE, FlashAttention-2 (через `torch.scaled_dot_product_attention`)
- **FFN**: Mixture-of-Experts — 8 SwiGLU-экспертов, top-2 роутинг, load-balancing aux loss
- RMSNorm, tied embeddings
- Словарь: 65 536-токенный SentencePiece BPE (byte-fallback, ChatML-спецтокены)
- ~165-170M активных параметров/токен, ~293M хранится всего (из-за MoE)

## Структура проекта

| Файл | Роль |
|---|---|
| `config.py` | единый источник истины: гиперпараметры, спецтокены, датасеты, профили железа |
| `model.py` | архитектура LunamiLM |
| `tokenizer_train.py` | обучение SentencePiece BPE токенизатора |
| `dataset.py` | стриминг + смешивание + упаковка датасетов HuggingFace, ChatML loss-маска |
| `train.py` | цикл обучения (AMP, grad-аккумуляция, чекпойнты, resume) |
| `chat.py` | инференс / REPL-чат |

## Данные

v1 — только **English + Code**, без русского и прочих языков. Смесь 50/50:

| Источник | Роль | Вес |
|---|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | English pretrain | 40% |
| [codeparrot-clean-train](https://huggingface.co/datasets/codeparrot/codeparrot-clean-train) | Python pretrain | 40% |
| [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | code instruct | 10% |
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | English chat | 10% |

Все источники публичные — без gated-доступа и HF-токена (см. `config.default_dataset_sources()`).

## Профили железа

Проект рассчитан на гибридную работу с Lightning.ai Studios — переключение одним флагом:

| Профиль | GPU | dtype | батч | особенности |
|---|---|---|---|---|
| `rocket` | A100 40GB | bf16 | 16×2 (eff. 32) | `torch.compile`, без grad-checkpointing |
| `workhorse` | T4 15GB | fp16 | 2×8 (eff. 16) | GradScaler, gradient checkpointing |

```bash
python train.py --profile rocket      # A100
python train.py --profile workhorse   # T4
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

1. **Обучить токенизатор** — положить English-текст и код в `tokenizer_corpus/`, затем:
   ```bash
   python tokenizer_train.py
   ```
   Создаёт `tokenizer/tokenizer.model` + `tokenizer/tokenizer.vocab`.

2. **Обучить модель** (данные стримятся прямо с HuggingFace, скачивать вручную не нужно):
   ```bash
   python train.py --profile workhorse   # или rocket
   ```
   Продолжить обучение: `python train.py --profile workhorse --resume checkpoints/step_5000.pt`

3. **Пообщаться с моделью**:
   ```bash
   python chat.py --checkpoint checkpoints/step_50000.pt
   ```

## Статус проекта

- [x] Архитектура реализована и проверена вживую (реальные forward/backward, конечный убывающий loss)
- [x] Токенизатор обучен на реальном корпусе 1.1GB EN+Code, словарь 65 536, round-trip проверен на коде/emoji/юникоде
- [x] Цикл обучения проверен end-to-end на реальных стримингованных данных + resume чекпойнтов (локально, урезанная архитектура)
- [ ] Полноценное обучение на полном масштабе (нужен Lightning.ai A100/T4 — локальная GPU разработки всего 4GB)

## Известные ограничения

- `model.py` не реализует инкрементальный KV-cache (параметр `use_cache` — заглушка) — генерация в `chat.py` идёт честным full-recompute. Настоящий кэш — будущая доработка, не входит в v1.
- Разработка велась на потребительской GPU (RTX 2050, 4GB) — этого достаточно для проверки всей проводки (токенизатор + данные + обучение), но недостаточно для полноценного претрейна на целевом масштабе модели.

## Лицензия

MIT — см. [LICENSE](LICENSE).
