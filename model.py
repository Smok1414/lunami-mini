# -*- coding: utf-8 -*-
"""
model.py — полная архитектура Lunami-Mini:
гибрид Mamba-2 SSM + GQA-Transformer + MoE (Mixture of Experts).

    Архитектура 2026 года:
        - 24 слоя, чередование [Mamba, Mamba, Mamba, Attention] × 6.
        - 18 слоёв Mamba-2 (быстрый SSM, d_state=64) + 6 слоёв GQA Attention (8Q/2KV, RoPE).
        - FFN только в Attention-блоках — как MoE: 8 экспертов, top-2 роутинг + load-balance лосс.
        - Активация SwiGLU, нормализация RMSNorm, эмбеддинги привязаны (tie), RoPE позиционка.
        - Контекст до 8192 токенов.

Файл спроектирован так, чтобы работать «из коробки» в Google Colab (T4, fp16).
Для максимальной скорости Mamba-2 можно подключить пакет mamba_ssm; здесь
дается чистая, самостоятельная (зависит только от torch) и корректная
реализация chunkwise selective scan, совместимая с autograd.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, SpecialTokens


# ═════════════════════════════════════════════════════════════════════════════
# 1. Базовые примитивы: RMSNorm, RoPE, SwiGLU.
# ═════════════════════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    """RMSNorm — нормализация по RMS (быстрее LayerNorm, как в Llama)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        # weight инициализируется единицами, eps защищает от деления на ноль.
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x^2 → mean по последней оси → rsqrt → умножить на x.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Считаем норму в float32 для устойчивости при fp16, затем возвращаем тип.
        out = self._norm(x.float()).type_as(x)
        return self.weight * out


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Предвычисление cos/sin таблиц для RoPE (как в Llama/GPT-NeoX).
    Возвращает (cos, sin) формы (1, 1, max_seq_len, head_dim).
    Форма именно (1, 1, T, hd): ось T стоит третьей, чтобы корректно
    бродкаститься против q/k формы (B, n_heads, T, head_dim) —
    (B, nh, T, hd) × (1, 1, T, hd) → совпадение по nh (vs 1) и по T (vs T).
    Раньше было (1, T, 1, hd) — это ставило T на ось nh и падало при T != n_heads.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)        # (T, head_dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    # Дублируем половину, чтобы получить полный head_dim (стиль GPT-NeoX/Llama).
    cos = torch.cat([cos, cos], dim=-1)[None, None, :, :]  # (1, 1, T, head_dim)
    sin = torch.cat([sin, sin], dim=-1)[None, None, :, :]
    return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Поворот вектора на 90°: (x1|x2) → (-x2|x1). Часть формулы RoPE."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Применение RoPE к тензору запроса q формы (B, n_heads, T, head_dim).
    cos/sin: (1, 1, max_seq_len, head_dim) — нарезаются по фактической длине T.
    """
    seq_len = q.shape[-2]
    cos = cos[:, :, :seq_len, :].to(q.dtype)  # (1, 1, T, head_dim)
    sin = sin[:, :, :seq_len, :].to(q.dtype)
    return q * cos + rotate_half(q) * sin


class SwiGLUMLP(nn.Module):
    """
    SwiGLU-FFN одного эксперта (или плотного пути).
        out = down( silu(gate(x)) * up(x) )
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        hidden = cfg.ffn_hidden
        self.w_gate = nn.Linear(d, hidden, bias=cfg.use_bias)
        self.w_up = nn.Linear(d, hidden, bias=cfg.use_bias)
        self.w_down = nn.Linear(hidden, d, bias=cfg.use_bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ═════════════════════════════════════════════════════════════════════════════
# 2. Mamba-2 SSM: selective scan (chunkwise) + блок.
# ═════════════════════════════════════════════════════════════════════════════
def ssd_sequential(
    X: torch.Tensor,
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """
    Naive (последовательный) selective scan — для проверки корректности на малом L.
    Все тензоры: (B, L, H, D). Рекурренция по времени (диагональный SSM):
        h_t = A_bar_t * h_{t-1} + B_bar_t * X_t
        y_t = C_t * h_t
    Возвращает y формы (B, L, H, D).

    NB: O(L) шагов во времени — медленно, но гарантированно корректно.
    """
    Bsz, L, H, D = X.shape
    h = torch.zeros(Bsz, H, D, device=X.device, dtype=X.dtype)
    ys: List[torch.Tensor] = []
    for t in range(L):
        h = A_bar[:, t] * h + B_bar[:, t] * X[:, t]    # (B,H,D)
        y = C[:, t] * h                                 # (B,H,D)
        ys.append(y)
    return torch.stack(ys, dim=1)                       # (B,L,H,D)


def ssd_chunked(
    X: torch.Tensor,
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    C: torch.Tensor,
    chunk: int = 256,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Нумерически-устойчивый chunkwise selective scan (SSD — State Space Duality, Mamba-2).

    Диагональный SSM-шаг:  h_t = A_bar_t · h_{t-1} + (B_bar_t · X_t);   y_t = C_t · h_t.

    Раскладывает рекурренцию на:
        1) intra-chunk — рекуррентный скан по t=0..chunk-1, ВЕКТОРИЗОВАННО по всем K=L/chunk
           чанкам одновременно (тензор (B,K,H,D)). Стартовое состояние 0 внутри каждого чанка.
        2) inter-chunk — K шагов рекурренции для приносимого состояния h_carry (малое число шагов).

    ПОЧЕМУ НЕ cumsum-трюк (term = (B*X)/G_t, y = C·G·cumsum): при s4d-linspace инициализации
    (|A| до 64, dt≈softplus(0)≈0.69) пер-шаговый A_bar достигает exp(0.69·−64)≈5.5e-20, и уже
    на ~3-м шаге внутри чанка накопленное G_t→0 (underflow даже в fp64). Тогда (B*X)/G_t = inf,
    cumsum = inf, и C·0·inf = NaN — модель выдавала NaN ещё на первом forward-проходе, до обучения.
    Fix: НИКОГДА НЕ ДЕЛИМ на накопленное произведение. Внутри-чанковый шаг — это обычная
    SSM-рекурренция A_bar_t·h_{t-1}+u_t (состояние остаётся ограниченным, произведение не накап-
    ливается ⇒ NaN-невозможно по построению). Вклад чужого состояния = C_t·h_carry·exp(log-префикс)
    — это МНОЖЕЖИТЕЛЬ (≤1, exp от неположительного), а не делитель; его underflow даёт ЧИСТЫЙ 0
    (= корректное экспоненциальное затухание быстро-распадающегося канала), без inf ⇒ без NaN.

    Тензоры входа: (B, L, H, D). Выход: (B, L, H, D).
    Память: O(B*L*H*D) на каждый временно́й буфер — как у обычных активаций.
    """
    Bsz, L, H, D = X.shape
    pad = (-L) % chunk
    if pad != 0:
        z = torch.zeros(Bsz, pad, H, D, device=X.device, dtype=X.dtype)
        # A_bar в пад-зоне = 1 (log 0 — нейтральный множитель), иначе log(0)→−inf испортит префикс.
        ones_pad = torch.ones(Bsz, pad, H, D, device=X.device, dtype=X.dtype)
        X = torch.cat([X, z], dim=1)
        A_bar = torch.cat([A_bar, ones_pad], dim=1)
        B_bar = torch.cat([B_bar, z], dim=1)
        C = torch.cat([C, z], dim=1)

    K = X.shape[1] // chunk
    Xr = X.view(Bsz, K, chunk, H, D)
    Ar = A_bar.view(Bsz, K, chunk, H, D)
    Br = B_bar.view(Bsz, K, chunk, H, D)
    Cr = C.view(Bsz, K, chunk, H, D)

    # ── 1) ВНУТРИ-чанковый скан (стартовое состояние 0 в каждом чанке, векторизовано по K) ───
    # h_local_t = A_bar_t · h_local_{t-1} + (B_bar_t · X_t) — рекуррентно по t=0..chunk-1
    # на тензоре (B,K,H,D): ВСЕ K чанков обрабатываются одновременно. ВАЖНО: накопленное
    # ПРОИЗВЕДЕНИЕ A_bar НЕ МАТЕРИАЛИЗУЕТСЯ — на каждой итерации лишь одно умножение на
    # A_bar_t ∈ (0,1]; состояние остаётся ограниченным ⇒ underflow/inf/NaN невозможны по построению.
    # h_local[:, k] хранит состояние в КОНЦЕ чанка k (нужно на этапе 2).
    h_local = torch.zeros(Bsz, K, H, D, device=X.device, dtype=X.dtype)
    y_local_chunks: List[torch.Tensor] = []
    for t in range(chunk):
        u_t = Br[:, :, t] * Xr[:, :, t]                # (B,K,H,D) — вход-заряд шага t
        h_local = Ar[:, :, t] * h_local + u_t          # SSM-шаг из стартового 0 (intra-chunk)
        y_local_chunks.append(Cr[:, :, t] * h_local)    # вклад внутри-чанковых входов в выход y_t
    Y_local = torch.stack(y_local_chunks, dim=2)         # (B,K,chunk,H,D)

    # ── 2) МЕЖ-чанковая рекурренция (входящее состояние h_carry) ─────────────────────────
    # Вклад чужого состояния в выход t чанка k:  C_t · h_carry · ∏_{i≤t} A_bar_i^{(k)}.
    # Считаем в лог-пространстве: log_prefix_incl_t = Σ_{i≤t} ln A_bar_i (cumsum, включит. шаг t).
    # carry_decay_t = exp(log_prefix_incl_t) ∈ (0,1] — это МНОЖИТЕЛЬ состояния, не делитель.
    # Underflow даёт ЧИСТЫЙ 0 (корректное экспоненциальное затухание агрессивно распадающегося
    # SSM-канала): умножение, без 0·inf ⇒ без NaN. Прежняя форма term=(B*X)/G с G_t→0 давала
    # inf → C·0·inf = NaN — мотивация замены (NaN порождался ещё на первом forward при s4d-init).
    log_prefix_incl = Ar.clamp_min(eps).log().cumsum(dim=2)   # (B,K,chunk,H,D): Σ_{i≤t} ln A_bar_i
    carry_decay = torch.exp(log_prefix_incl)                  # (B,K,chunk,H,D), ∈ (0,1]

    h_carry = torch.zeros(Bsz, H, D, device=X.device, dtype=X.dtype)
    inter_chunks: List[torch.Tensor] = []
    for k in range(K):
        # выход чанка k = выход внутри-чанковых входов (Y_local) + вклад приносимого состояния.
        inter = Cr[:, k] * carry_decay[:, k] * h_carry.unsqueeze(1)   # (B,chunk,H,D)
        inter_chunks.append(Y_local[:, k] + inter)
        # состояние в конце чанка k = h_carry·(∏ всего чанка A) + h_local_end_k → вход в чанк k+1.
        G_total_k = carry_decay[:, k, -1]                   # ∏_{i=0..chunk-1} A_bar_i^{(k)} (может →0)
        h_carry = G_total_k * h_carry + h_local[:, k]        # стартовое состояние чанка k+1

    Y = torch.stack(inter_chunks, dim=1)                  # (B,K,chunk,H,D)
    Y = Y.reshape(Bsz, K * chunk, H, D)                   # склеиваем чанки
    Y = Y[:, :L, :]                                       # отбрасываем паддинг
    return Y


class Mamba2Block(nn.Module):
    """
    Блок Mamba-2 SSM.
        in_proj  : d_model → 2*d_inner (вход x + гейт z)
        conv1d   : короткая causal-конволюция (ядро 4) + SiLU
        x_proj   : d_inner → (dt, B, C)      — input-dependent параметры SSM
        dt_proj  : уточнение шага Δ через SiLU+Linear
        ssd_scan : selective scan (SSD) → y
        skip     : y = C·S + D·x  (D — residual-skip параметр)
        out_proj : d_inner → d_model
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        d_inner = cfg.mamba_expand * d                    # 1536
        D = cfg.mamba_d_state                              # 64 (head_dim / длина состояния)
        H = d_inner // D                                   # 24 "головы" SSD
        self.d_model = d
        self.d_inner = d_inner
        self.H = H
        self.D = D
        self.chunk = cfg.mamba_chunk_size
        self.use_fast = cfg.mamba_use_fast

        self.in_proj = nn.Linear(d, 2 * d_inner, bias=cfg.use_bias)
        # Групповая conv1d (depthwise) — независимый фильтр на каждый канал.
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=cfg.mamba_conv_kernel,
            padding=cfg.mamba_conv_kernel - 1,
            groups=d_inner,
            bias=True,
        )
        # x_proj выход: dt (H) + B_t (D) + C_t (D). dt — на «голову», B/C — общие на головы.
        self.x_proj = nn.Linear(d_inner, H + 2 * D, bias=False)
        self.dt_proj = nn.Linear(H, H, bias=True)

        # A_log — лог положительной матрицы перехода A (H, D). Инициализация s4d-linspace.
        A_init = torch.arange(1, D + 1, dtype=torch.float32).repeat(H, 1)  # (H, D)
        self.A_log = nn.Parameter(torch.log(A_init))
        # D — параметр skip-связи на голову (скаляр).
        self.D_param = nn.Parameter(torch.ones(H))

        self.out_proj = nn.Linear(d_inner, d, bias=cfg.use_bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, d_model) — уже нормализованный вход. → (B, L, d_model)."""
        Bsz, L, _ = u.shape
        H, D, d_inner = self.H, self.D, self.d_inner

        # 1) in_proj → x, z.
        xz = self.in_proj(u)                              # (B, L, 2*d_inner)
        x, z = xz.split(d_inner, dim=-1)                # (B, L, d_inner)

        # 2) causal short conv + SiLU.
        x = x.transpose(1, 2)                            # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]                     # трим будущие токены → causal
        x = x.transpose(1, 2)                            # (B, L, d_inner)
        x = F.silu(x)

        # 3) dt, B_t, C_t из x (input-dependent параметры SSM).
        x_dbl = self.x_proj(x)                           # (B, L, H + 2D)
        dt, B_t, C_t = x_dbl.split([H, D, D], dim=-1)    # dt:(B,L,H)  B_t,C_t:(B,L,D)
        dt = F.softplus(self.dt_proj(dt))                # Δ > 0
        A = -torch.exp(self.A_log)                       # (H, D) — отрицательный для затухания

        # Переход в head-вид: (B, L, H, D).
        x_h = x.view(Bsz, L, H, D)
        dt_e = dt.unsqueeze(-1)                          # (B, L, H, 1)
        A_exp = A.view(1, 1, H, D)
        A_bar = torch.exp(dt_e * A_exp)                   # (B, L, H, D)
        B_bar = dt_e * B_t.view(Bsz, L, 1, D)             # (B, L, H, D)
        C_t_full = C_t.view(Bsz, L, 1, D).expand(Bsz, L, H, D)

        # 4) Selective scan (в fp32 для точности, как в референс-реализации Mamba).
        orig_dtype = x_h.dtype
        scan_args = (x_h.float(), A_bar.float(), B_bar.float(), C_t_full.float())
        if self.use_fast:
            Y = ssd_chunked(*scan_args, chunk=self.chunk)
        else:
            Y = ssd_sequential(*scan_args)
        Y = Y.to(orig_dtype)                             # (B, L, H, D)

        # 5) Skip-связь D·x, gating по z (SiLU), out_proj.
        Y = Y + self.D_param.view(1, 1, H, 1) * x_h      # (B, L, H, D)
        y = Y.reshape(Bsz, L, d_inner)                    # (B, L, d_inner)
        y = y * F.silu(z)                                 # gating ветки z
        return self.out_proj(y)                           # (B, L, d)


class MambaLayer(nn.Module):
    """Полный residual-слой: pre-norm → Mamba-2 → residual + dropout."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.block = Mamba2Block(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x + self.dropout(self.block(self.norm(x))), torch.zeros((), device=x.device, dtype=x.dtype)


# ═════════════════════════════════════════════════════════════════════════════
# 3. GQA Attention + MoE FFN.
# ═════════════════════════════════════════════════════════════════════════════
class GQAAttention(nn.Module):
    """Grouped-Query Attention (как в Llama 3) с RoPE и опциональным FlashAttention."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.n_heads_q = cfg.n_heads_q
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim_computed
        self.n_groups = self.n_heads_q // self.n_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_bias = cfg.use_bias

        self.q_proj = nn.Linear(d, self.n_heads_q * self.head_dim, bias=cfg.use_bias)
        self.kv_proj = nn.Linear(d, 2 * self.n_kv_heads * self.head_dim, bias=cfg.use_bias)
        self.o_proj = nn.Linear(self.n_heads_q * self.head_dim, d, bias=cfg.use_bias)

        # Регистрируем предвычисленные cos/sin для RoPE-позиционки как буферы.
        cos, sin = precompute_rope(self.head_dim, cfg.max_seq_len, theta=cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.flash = cfg.flash_attention and self._flash_available()

    @staticmethod
    def _flash_available() -> bool:
        """Проверяем наличие ускоренного fused attention (torch SDPA/FA2)."""
        try:
            _ = F.scaled_dot_product_attention
            return True
        except AttributeError:
            return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, T, _ = x.shape
        q = self.q_proj(x).view(Bsz, T, self.n_heads_q, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(Bsz, T, 2, self.n_kv_heads, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)                  # (B, n_kv, T, hd)
        v = kv[:, :, 1].transpose(1, 2)

        # RoPE применяется к q и k (v	position не кодируется).
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # GQA: реплицируем KV-головы до числа query-голов.
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        if self.flash:
            # F.scaled_dot_product_attention с is_causal=True — fused kernel (FlashAttention-2 путь).
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)
            # Кausal-маска: верхний треугольник → -inf.
            mask = torch.full((T, T), float("-inf"), device=x.device, dtype=scores.dtype)
            mask = torch.triu(mask, diagonal=1)
            scores = scores + mask
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(Bsz, T, self.n_heads_q * self.head_dim)
        return self.o_proj(out)


class MoELayer(nn.Module):
    """
    Mixture-of-Experts (MoE) FFN с top-k routing.
        - n_experts экспертов (SwiGLU), n_active_experts активных на токен.
        - Load-balancing aux loss (Switch-Transformer style):
              aux = E * Σ_e f_e * P_e
          где f_e — доля токенов, отрoутленных к эксперту e, P_e — средняя вероятность.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.n_experts = cfg.n_experts
        self.n_active = cfg.n_active_experts
        self.aux_weight = cfg.moe_aux_loss_weight
        self.noise_std = cfg.moe_noise_std

        self.router = nn.Linear(d, self.n_experts, bias=cfg.moe_router_bias)
        self.experts = nn.ModuleList([SwiGLUMLP(cfg) for _ in range(self.n_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Bsz, T, d = x.shape
        N = Bsz * T
        flat = x.reshape(N, d)

        logits = self.router(flat)                       # (N, E)
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # Top-k routing (k = n_active): выбираем экспертов с большим логитом.
        topk_logits, topk_idx = torch.topk(logits, self.n_active, dim=-1)  # (N, k)
        topk_weights = F.softmax(topk_logits, dim=-1)    # нормировка весов по выбранным экспертам

        # Диспетчеризация: для каждого эксперта собираем свои токены и складываем результаты.
        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            # Маска токенов, у которых эксперт e попал в top-k.
            mask = (topk_idx == e).any(dim=-1)           # (N,)
            if not mask.any():
                continue
            # Суммарный вес эксперта e для каждого подходящего токена.
            exp_weight = (topk_weights * (topk_idx == e).float()).sum(dim=-1)[mask]
            y_e = self.experts[e](flat[mask])            # (M, d)
            out[mask] += y_e * exp_weight.unsqueeze(-1)

        # ── Вспомогательный loss балансировки нагрузки (Switch-Transformer style) ─
        # NB: P обязан сохранять градиент (через router logits), иначе у aux_loss
        # не будет производной по весам роутера и балансировка МОЛЧАЛИВО не работает.
        # f — НЕ дифференцируем (счёт по аргмаксу из topk), его считаем под no_grad.
        probs = F.softmax(logits, dim=-1)                 # (N, E) — дифференцируемый
        with torch.no_grad():
            one_hot = torch.zeros(N, self.n_active, self.n_experts, device=x.device, dtype=logits.dtype)
            one_hot.scatter_(2, topk_idx.unsqueeze(-1), 1.0)
            f = one_hot.sum(dim=1).mean(dim=0)             # (E,) — без градиента (доля токенов)
        P = probs.mean(dim=0)                              # (E,) — с градиентом
        aux = self.n_experts * (f * P).sum()               # градиент течёт через P → router
        aux_loss = self.aux_weight * aux

        return out.view_as(x), aux_loss


class AttentionLayer(nn.Module):
    """Полный Attention-слой: pre-norm → GQA → residual → pre-norm → MoE FFN → residual."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn: nn.Module = MoELayer(cfg) if cfg.use_moe else SwiGLUMLP(cfg)
        self.use_moe = cfg.use_moe
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.attn(self.norm1(x)))
        f = self.norm2(x)
        if self.use_moe:
            out, aux = self.ffn(f)
            return x + self.dropout(out), aux
        return x + self.dropout(self.ffn(f)), torch.zeros((), device=x.device, dtype=x.dtype)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Полная модель Lunami-Mini.
# ═════════════════════════════════════════════════════════════════════════════
class LunamiLM(nn.Module):
    """
    Гибрид Mamba-2 + GQA + MoE: токен-эмбеддинги → [M,M,M,A]×6 слоёв → RMSNorm → lm_head.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.layers = nn.ModuleList()
        for layer_type in cfg.layer_pattern:
            if layer_type == 0:
                self.layers.append(MambaLayer(cfg))
            elif layer_type == 1:
                self.layers.append(AttentionLayer(cfg))
            else:
                raise ValueError(f"Неизвестный тип блока в layer_pattern: {layer_type}")
        self.norm_f = RMSNorm(d, cfg.norm_eps)
        # lm_head создаём всегда; при tie_embeddings шаррим веса с эмбеддингом.
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Инициализация весов в стиле Llama/Mamba."""
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Прямой проход.
            input_ids: (B, L) — ID токенов.
            labels: (B, L) — те же ID со смещением, с IGNORE_INDEX на токенах, loss по которым не считаем.
        Возвращает словарь {'logits', 'loss' (или None), 'aux_loss'}.
        """
        Bsz, L = input_ids.shape
        x = self.tok_emb(input_ids)                      # (B, L, d)
        aux_total = torch.zeros((), device=input_ids.device, dtype=x.dtype)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # recomputation активаций при backward (экономия памяти).
                x, aux = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False
                )
            else:
                x, aux = layer(x)
            aux_total = aux_total + aux

        x = self.norm_f(x)
        logits = self.lm_head(x)                         # (B, L, V)

        loss = None
        if labels is not None:
            # Сдвиг на 1: предсказываем следующий токен.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=SpecialTokens.IGNORE_INDEX,
            )
            loss = loss + aux_total.to(loss.dtype)       # добавляем MoE load-balance loss

        return {"logits": logits, "loss": loss, "aux_loss": aux_total}

    @torch.no_grad()
    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not only_trainable))

    def summary(self) -> str:
        """Краткое описание модели для логирования."""
        total = self.num_parameters(only_trainable=False)
        params = count_parameters_real(self.cfg)
        return (
            f"LunamiLM | d_model={self.cfg.d_model} слоёв={self.cfg.n_layers} "
            f"(M={self.cfg.n_mamba}, A={self.cfg.n_attention}) MoE={self.cfg.n_experts}×{self.cfg.n_active_experts}\n"
            f"  параметров (PyTorch): {total:,} ({total/1e6:.1f}M)\n"
            f"  эмбеддинги: {params[0]:,} | Mamba: {params[1]:,} | Attn: {params[2]:,} | MoE(all): {params[3]:,}\n"
            f"  tie_embeddings={self.cfg.tie_embeddings} | контекст={self.cfg.max_seq_len}"
        )


def count_parameters_real(cfg: ModelConfig) -> Tuple[int, int, int, int]:
    """Быстрая оценка параметров по группам (эмбеддинги / Mamba / Attn / MoE)."""
    d = cfg.d_model
    D = cfg.mamba_d_state
    d_inner = cfg.mamba_expand * d
    H = d_inner // D
    head_dim = cfg.head_dim_computed

    embed = cfg.vocab_size * d if cfg.tie_embeddings else 2 * cfg.vocab_size * d

    # Mamba-2 (per layer):
    mamba_per = (
        d * (2 * d_inner)                                  # in_proj
        + d_inner * cfg.mamba_conv_kernel                  # conv1d (depthwise)
        + d_inner * (H + 2 * D)                            # x_proj
        + H * H                                            # dt_proj
        + H * D                                            # A_log
        + H                                                # D_param
        + d_inner * d                                      # out_proj
    )
    mamba_total = mamba_per * cfg.n_mamba

    # Attention (GQA) per layer:
    attn_per = (
        d * (cfg.n_heads_q * head_dim)                     # q_proj
        + d * (2 * cfg.n_kv_heads * head_dim)              # kv_proj
        + (cfg.n_heads_q * head_dim) * d                  # o_proj (RoPE без обучаемых параметров)
    )
    attn_total = attn_per * cfg.n_attention

    # MoE FFN: SwiGLU эксперт = 3*d*hidden; роутер = d*n_experts; × на число Attention слоёв.
    hidden = cfg.ffn_hidden
    expert = 3 * d * hidden
    if cfg.use_moe:
        moe_total = (expert * cfg.n_experts + d * cfg.n_experts) * cfg.n_attention
    else:
        moe_total = expert * cfg.n_attention

    return embed, mamba_total, attn_total, moe_total


if __name__ == "__main__":
    # Быстрый smoke-test: создаём модель маленького размера и прогоняем forward.
    cfg = ModelConfig(d_model=128, max_seq_len=256, vocab_size=512, n_layers=4, n_mamba=3, n_attention=1,
                      layer_pattern=(0, 0, 0, 1), n_experts=4, n_active_experts=2,
                      n_heads_q=4, n_kv_heads=2, ffn_expand=2)
    model = LunamiLM(cfg)
    print(model.summary())
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    labels = ids.clone()
    out = model(ids, labels=labels)
    print("logits:", out["logits"].shape, "| loss:", float(out["loss"]) if out["loss"] is not None else None)
