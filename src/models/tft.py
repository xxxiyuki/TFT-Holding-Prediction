"""From-scratch Temporal Fusion Transformer (TFT).

Implements the building blocks described in Lim et al. (2021):

* Gated Residual Network (GRN) -- the universal non-linear unit.
* Variable Selection Network (VSN) -- soft feature selection over a
  set of inputs at each time step (or once for static covariates).
* LSTM encoder/decoder with locality-preserving recurrence.
* Static covariate encoders that condition the LSTM, the
  enrichment GRN, and the post-attention gating.
* Interpretable multi-head self-attention over the full sequence.
* Quantile output head producing one value per quantile.

The model is intentionally compact and CPU-friendly. It does not depend
on ``pytorch-lightning`` or ``pytorch-forecasting`` -- only ``torch``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class GatedLinearUnit(nn.Module):
    """GLU: ``sigmoid(Wx + b) * (Vx + c)``."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        a, b = self.fc(x).chunk(2, dim=-1)
        return torch.sigmoid(a) * b


class GatedResidualNetwork(nn.Module):
    """GRN with optional context vector and residual connection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        context_size: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        output_size = output_size or hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context = (
            nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        )
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GatedLinearUnit(hidden_size, output_size, dropout=dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip = (
            nn.Linear(input_size, output_size) if input_size != output_size else None
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = self.skip(x) if self.skip is not None else x
        h = self.fc1(x)
        if self.context is not None:
            if context is None:
                raise ValueError("GRN expected a context vector but got None")
            ctx = self.context(context)
            # broadcast context across the time dimension if needed
            while ctx.dim() < h.dim():
                ctx = ctx.unsqueeze(-2)
            h = h + ctx
        h = self.elu(h)
        h = self.fc2(h)
        h = self.glu(h)
        return self.layer_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """Soft selection over ``num_inputs`` features at each time step.

    Each input feature is first projected to ``hidden_size`` by its own
    GRN. A flat GRN over the concatenation of all features (optionally
    conditioned on a static context) produces softmax weights, and the
    output is the weighted sum of the per-feature embeddings.
    """

    def __init__(
        self,
        num_inputs: int,
        hidden_size: int,
        context_size: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.flatten_grn = GatedResidualNetwork(
            input_size=num_inputs * hidden_size,
            hidden_size=hidden_size,
            output_size=num_inputs,
            context_size=context_size,
            dropout=dropout,
        )
        self.per_feature_grn = nn.ModuleList(
            GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
            for _ in range(num_inputs)
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` shape: ``(..., num_inputs, hidden_size)``."""
        if x.size(-2) != self.num_inputs or x.size(-1) != self.hidden_size:
            raise ValueError(
                f"VSN got shape {tuple(x.shape)}, expected last two dims "
                f"({self.num_inputs}, {self.hidden_size})"
            )
        flat = x.flatten(start_dim=-2)
        weights = torch.softmax(self.flatten_grn(flat, context), dim=-1)
        weights = weights.unsqueeze(-1)  # (..., num_inputs, 1)
        processed = torch.stack(
            [grn(x[..., i, :]) for i, grn in enumerate(self.per_feature_grn)],
            dim=-2,
        )
        combined = (weights * processed).sum(dim=-2)
        return combined, weights.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention sharing the value projection across heads."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.ModuleList(
            nn.Linear(hidden_size, self.head_dim, bias=False) for _ in range(num_heads)
        )
        self.k_proj = nn.ModuleList(
            nn.Linear(hidden_size, self.head_dim, bias=False) for _ in range(num_heads)
        )
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=False)  # shared
        self.out_proj = nn.Linear(self.head_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_shared = self.v_proj(v)
        head_outputs = []
        attn_weights = []
        for h in range(self.num_heads):
            qh = self.q_proj[h](q)
            kh = self.k_proj[h](k)
            scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            head_outputs.append(torch.matmul(attn, v_shared))
            attn_weights.append(attn)
        # average across heads (interpretable variant)
        out = torch.stack(head_outputs, dim=0).mean(dim=0)
        weights = torch.stack(attn_weights, dim=0).mean(dim=0)
        return self.out_proj(out), weights


class AddNormGate(nn.Module):
    """GLU + residual + LayerNorm."""

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.glu = GatedLinearUnit(hidden_size, hidden_size, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(self.glu(x) + residual)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
@dataclass
class TFTConfig:
    num_funds: int                 # vocab size for fund_code (incl. unknown=0)
    num_stocks: int                # vocab size for stock_code
    num_static_numeric: int        # number of static numeric features
    num_encoder_features: int      # number of time-varying encoder features
    decoder_length: int = 1        # forecast horizon (in half-years)
    hidden_size: int = 32
    embedding_dim: int = 8
    lstm_layers: int = 1
    attention_heads: int = 2
    dropout: float = 0.1
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


class TemporalFusionTransformer(nn.Module):
    """Compact TFT for half-yearly fund-holding regression."""

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size

        # --- categorical embeddings + numeric projections ---
        self.fund_emb = nn.Embedding(config.num_funds, config.embedding_dim, padding_idx=0)
        self.stock_emb = nn.Embedding(config.num_stocks, config.embedding_dim, padding_idx=0)
        self.fund_proj = nn.Linear(config.embedding_dim, h)
        self.stock_proj = nn.Linear(config.embedding_dim, h)
        self.static_numeric_proj = nn.ModuleList(
            nn.Linear(1, h) for _ in range(config.num_static_numeric)
        )
        self.encoder_proj = nn.ModuleList(
            nn.Linear(1, h) for _ in range(config.num_encoder_features)
        )

        # --- variable selection ---
        num_static_inputs = 2 + config.num_static_numeric  # fund, stock, + numerics
        self.static_vsn = VariableSelectionNetwork(
            num_inputs=num_static_inputs,
            hidden_size=h,
            dropout=config.dropout,
        )
        self.encoder_vsn = VariableSelectionNetwork(
            num_inputs=config.num_encoder_features,
            hidden_size=h,
            context_size=h,
            dropout=config.dropout,
        )
        # Decoder VSN reuses the same encoder features (broadcast current
        # static state forward); for a 1-step forecast this is sufficient.
        self.decoder_vsn = VariableSelectionNetwork(
            num_inputs=config.num_encoder_features,
            hidden_size=h,
            context_size=h,
            dropout=config.dropout,
        )

        # --- static contexts (Lim et al. produce four) ---
        self.ctx_variable_selection = GatedResidualNetwork(h, h, dropout=config.dropout)
        self.ctx_lstm_h = GatedResidualNetwork(h, h, dropout=config.dropout)
        self.ctx_lstm_c = GatedResidualNetwork(h, h, dropout=config.dropout)
        self.ctx_enrichment = GatedResidualNetwork(h, h, dropout=config.dropout)

        # --- LSTM encoder / decoder ---
        self.lstm_encoder = nn.LSTM(
            input_size=h,
            hidden_size=h,
            num_layers=config.lstm_layers,
            batch_first=True,
        )
        self.lstm_decoder = nn.LSTM(
            input_size=h,
            hidden_size=h,
            num_layers=config.lstm_layers,
            batch_first=True,
        )
        self.post_lstm_gate = AddNormGate(h, dropout=config.dropout)

        # --- static enrichment + self-attention ---
        self.enrichment_grn = GatedResidualNetwork(
            h, h, context_size=h, dropout=config.dropout
        )
        self.self_attn = InterpretableMultiHeadAttention(
            h, num_heads=config.attention_heads, dropout=config.dropout
        )
        self.post_attn_gate = AddNormGate(h, dropout=config.dropout)
        self.position_wise_grn = GatedResidualNetwork(h, h, dropout=config.dropout)
        self.pre_output_gate = AddNormGate(h, dropout=config.dropout)

        # --- quantile output head ---
        self.output_layer = nn.Linear(h, len(config.quantiles))

    # ------------------------------------------------------------------ #
    def _project_inputs(
        self,
        fund_ids: torch.Tensor,
        stock_ids: torch.Tensor,
        static_numeric: torch.Tensor,
        encoder_numeric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(static_inputs, encoder_inputs)`` ready for VSNs."""
        fund_e = self.fund_proj(self.fund_emb(fund_ids))            # (B, H)
        stock_e = self.stock_proj(self.stock_emb(stock_ids))        # (B, H)

        static_pieces = [fund_e.unsqueeze(1), stock_e.unsqueeze(1)]  # (B, 1, H) each
        for i, proj in enumerate(self.static_numeric_proj):
            static_pieces.append(proj(static_numeric[:, i : i + 1]).unsqueeze(1))
        static_inputs = torch.cat(static_pieces, dim=1)              # (B, S, H)

        # encoder: (B, T, F_enc) -> (B, T, F_enc, H)
        b, t, f_enc = encoder_numeric.shape
        enc_pieces = []
        for i, proj in enumerate(self.encoder_proj):
            enc_pieces.append(proj(encoder_numeric[:, :, i : i + 1]).unsqueeze(2))
        encoder_inputs = torch.cat(enc_pieces, dim=2)                # (B, T, F_enc, H)
        assert encoder_inputs.shape == (b, t, f_enc, self.config.hidden_size)
        return static_inputs, encoder_inputs

    # ------------------------------------------------------------------ #
    def forward(
        self,
        fund_ids: torch.Tensor,         # (B,)
        stock_ids: torch.Tensor,        # (B,)
        static_numeric: torch.Tensor,   # (B, F_stat)
        encoder_numeric: torch.Tensor,  # (B, T, F_enc)
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        b, t, _ = encoder_numeric.shape

        static_inputs, encoder_inputs = self._project_inputs(
            fund_ids, stock_ids, static_numeric, encoder_numeric
        )

        # --- static covariate encoding ---
        static_embedding, static_weights = self.static_vsn(static_inputs)  # (B, H)
        ctx_vs = self.ctx_variable_selection(static_embedding)
        ctx_h0 = self.ctx_lstm_h(static_embedding)
        ctx_c0 = self.ctx_lstm_c(static_embedding)
        ctx_enr = self.ctx_enrichment(static_embedding)

        # --- encoder VSN (per time step, conditioned on ctx_vs) ---
        ctx_vs_seq = ctx_vs.unsqueeze(1).expand(-1, t, -1)
        encoder_vsn_out, encoder_var_weights = self.encoder_vsn(
            encoder_inputs, ctx_vs_seq
        )                                                                  # (B, T, H)

        # --- decoder VSN (broadcast last encoder step into the future) ---
        last_step = encoder_inputs[:, -1:, :, :].expand(
            -1, cfg.decoder_length, -1, -1
        )
        ctx_vs_dec = ctx_vs.unsqueeze(1).expand(-1, cfg.decoder_length, -1)
        decoder_vsn_out, decoder_var_weights = self.decoder_vsn(
            last_step, ctx_vs_dec
        )                                                                  # (B, D, H)

        # --- LSTM encoder/decoder seeded by static context ---
        h0 = ctx_h0.unsqueeze(0).expand(cfg.lstm_layers, -1, -1).contiguous()
        c0 = ctx_c0.unsqueeze(0).expand(cfg.lstm_layers, -1, -1).contiguous()
        enc_out, (h_n, c_n) = self.lstm_encoder(encoder_vsn_out, (h0, c0))
        dec_out, _ = self.lstm_decoder(decoder_vsn_out, (h_n, c_n))

        lstm_out = torch.cat([enc_out, dec_out], dim=1)                     # (B, T+D, H)
        vsn_concat = torch.cat([encoder_vsn_out, decoder_vsn_out], dim=1)
        gated = self.post_lstm_gate(lstm_out, vsn_concat)

        # --- static enrichment ---
        enriched = self.enrichment_grn(gated, ctx_enr)

        # --- causal self-attention only over the decoder steps ---
        seq_len = enriched.size(1)
        # mask shape (1, D, T+D): decoder query attends to all encoder
        # steps and to itself causally.
        causal = torch.ones(cfg.decoder_length, seq_len, device=enriched.device)
        for i in range(cfg.decoder_length):
            causal[i, t + i + 1 :] = 0
        causal = causal.unsqueeze(0)
        decoder_query = enriched[:, t:, :]
        attn_out, attn_weights = self.self_attn(
            decoder_query, enriched, enriched, mask=causal
        )
        attn_out = self.post_attn_gate(attn_out, decoder_query)

        # --- position-wise feed-forward + final gate ---
        ff = self.position_wise_grn(attn_out)
        out = self.pre_output_gate(ff, gated[:, t:, :])

        # --- quantile head: take the (only) decoder step ---
        # out: (B, D, H) -> (B, n_quantiles) for D=1
        last = out[:, -1, :]
        quantile_pred = self.output_layer(last)
        return {
            "quantile_pred": quantile_pred,                # (B, n_quantiles)
            "static_var_weights": static_weights,           # (B, num_static)
            "encoder_var_weights": encoder_var_weights,     # (B, T, F_enc)
            "decoder_var_weights": decoder_var_weights,     # (B, D, F_enc)
            "attention_weights": attn_weights,              # (B, D, T+D)
        }
