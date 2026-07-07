# Tokenizer

SentencePiece BPE, vocabulary size 65,536, trained on a 1.1GB English + Code corpus (576.8MB FineWeb-Edu, 576.8MB Python from codeparrot-clean).

## Design decisions

Before training the tokenizer, the design was checked against how modern tokenizers handle these same tradeoffs (Llama 3, Qwen, DeepSeek, Gemma, Mistral, SmolLM2, Phi-4, GPT). The general direction across all of them is byte-level BPE, mainly for two reasons: it avoids `<unk>` entirely, and it preserves whitespace exactly. SentencePiece was the required library here, so the same behavior is approximated with these flags:

- `byte_fallback=True` — an out-of-vocabulary character is encoded as its raw UTF-8 bytes instead of `<unk>`.
- `normalization="identity"` (not NFKC) — SentencePiece's built-in NFKC variant collapses whitespace runs, which destroys Python indentation. Identity normalization keeps bytes exact.
- `add_dummy_prefix=False` — no leading space inserted, so encoding is not sensitive to whether a string starts with a space.
- `remove_extra_whitespaces=False` — indentation and repeated spaces are preserved rather than collapsed.
- `split_digits=True` — each digit is its own token, which generalizes better for numbers and code literals.

## Round-trip check

```python
>>> sp.encode("def f():\n\treturn 1\n")
['def', '▁f', '(', ')', ':', '<0x0A>', '<0x09>', 're', 't', 'u', 'r', 'n', '▁1', '<0x0A>']
>>> sp.decode(sp.encode(text)) == text
True   # tabs and newlines survive, byte-for-byte
```

Also checked: emoji, accented Unicode, long documents, and unknown byte sequences — all round-trip exactly (see `tokenizer_train.py`'s built-in verification step, which runs automatically after training and fails loudly if any of this breaks).

## Special tokens

ChatML-style, IDs fixed in `config.SpecialTokens`:

| ID | Token | Role |
|---|---|---|
| 0 | `<\|pad\|>` | padding |
| 1 | `<\|bos\|>` | beginning of sequence |
| 2 | `<\|eos\|>` | end of sequence |
| 3 | `<\|unk\|>` | unknown (required by SentencePiece, effectively unused due to byte-fallback) |
| 4 | `<\|system\|>` | system turn marker |
| 5 | `<\|user\|>` | user turn marker |
| 6 | `<\|assistant\|>` | assistant turn marker (loss is computed on tokens after this) |
| 7 | `<\|end\|>` | end of a turn |
| 8 | `<\|code\|>` | start of a code block |
| 9 | `<\|endofcode\|>` | end of a code block |

`dataset.py` loads the tokenizer and asserts these IDs match before training starts, so a mismatch fails immediately instead of silently corrupting the loss mask.
