# Architecture

Current implementation: 24 layers, repeating `[Mamba-2 ×3 → GQA Attention + MoE ×1]` six times.

```mermaid
flowchart TB
    input(["Token IDs"]) --> emb["Token Embedding<br/>vocab = 65,536"]
    emb --> stack

    subgraph stack [" × 6 "]
        direction LR
        m1["Mamba-2"] --> m2["Mamba-2"] --> m3["Mamba-2"] --> att["GQA Attention<br/>+ MoE FFN"]
    end

    stack --> norm["Final RMSNorm"]
    norm --> head["LM Head<br/>(tied embedding)"]
    head --> logits(["Logits<br/>vocab = 65,536"])

    style m1 fill:#8A2BE2,color:#fff,stroke:#5a1a99
    style m2 fill:#8A2BE2,color:#fff,stroke:#5a1a99
    style m3 fill:#8A2BE2,color:#fff,stroke:#5a1a99
    style att fill:#FF8C00,color:#fff,stroke:#b35f00
    style emb fill:#1f6feb,color:#fff
    style head fill:#1f6feb,color:#fff
```

## Mamba-2 block (18 of 24 layers)

```mermaid
flowchart TB
    x["input x"] --> rn["RMSNorm"]
    rn --> inproj["in_proj<br/>d_model → 2×d_inner"]
    inproj --> xbranch["x branch"]
    inproj --> zbranch["z branch (gate)"]
    xbranch --> conv["Causal Conv1d<br/>kernel=4, depthwise"]
    conv --> silu["SiLU"]
    silu --> xproj["x_proj → Δ, B, C"]
    xproj --> scan["Selective Scan (SSD)<br/>d_state = 64, chunked"]
    scan --> skip["+ D · x  (skip)"]
    skip --> gatemul["× SiLU(z)"]
    zbranch --> gatemul
    gatemul --> outproj["out_proj<br/>d_inner → d_model"]
    outproj --> add(("+"))
    x --> add
    add --> y["output"]

    style scan fill:#8A2BE2,color:#fff
```

The selective scan is implemented as a chunkwise algorithm (SSD-style) in plain PyTorch — no `mamba_ssm` dependency required to run. An optional fused backend (`mamba_ssm`) can be used instead when installed; see [engineering.md](engineering.md#fast-mamba-backend).

## GQA + MoE block (6 of 24 layers)

```mermaid
flowchart TB
    x2["input x"] --> rn1["RMSNorm"]
    rn1 --> gqa["GQA Attention<br/>8 Q heads / 2 KV heads<br/>RoPE + FlashAttention-2"]
    gqa --> add1(("+"))
    x2 --> add1
    add1 --> rn2["RMSNorm"]
    rn2 --> router["Router: top-2 of 8"]
    router --> experts["8 × SwiGLU experts<br/>(2 active per token)"]
    experts --> weighted["weighted sum"]
    weighted --> add2(("+"))
    add1 --> add2
    add2 --> y2["output"]

    style gqa fill:#FF8C00,color:#fff
    style router fill:#FF8C00,color:#fff
```

## Component summary

| Component | Spec | Notes |
|---|---|---|
| Mamba-2 blocks (18/24) | `d_state=64`, `expand=2`, `conv_kernel=4` | Chunkwise selective scan (SSD), pure PyTorch, autograd-safe |
| GQA attention (6/24) | 8 query heads / 2 KV heads, `head_dim=96` | RoPE, fused SDPA (FlashAttention-2 path) |
| FFN | Mixture-of-Experts, 8×SwiGLU, top-2 routing | Load-balancing aux loss, gradient flows through the router |
| Norm / embeddings | RMSNorm, tied input/output | `d_model=768`, context up to 8,192 tokens (see [training.md](training.md) for what's actually been run) |
| Tokenizer | SentencePiece BPE, 65,536 vocab | See [tokenizer.md](tokenizer.md) |
| Scale | ~165.5M active parameters/token, ~293M total stored | Sparse MoE — active/total split similar in spirit to Mixtral, at a much smaller scale |

Parameter counts are measured directly from `model.num_parameters()` on the instantiated model, not estimated — see `count_parameters()` / `count_parameters_real()` in `config.py` / `model.py` for the breakdown by component.
