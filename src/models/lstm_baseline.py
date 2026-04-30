"""Simple LSTM encoder-decoder baseline matching the TFT input contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LSTMConfig:
    num_funds: int
    num_stocks: int
    num_static_numeric: int
    num_encoder_features: int
    decoder_length: int = 1
    hidden_size: int = 64
    embedding_dim: int = 8
    num_layers: int = 1
    dropout: float = 0.1
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


class LSTMBaseline(nn.Module):
    """Embeddings + LSTM encoder + 1-step decoder + quantile head."""

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.fund_emb = nn.Embedding(config.num_funds, config.embedding_dim, padding_idx=0)
        self.stock_emb = nn.Embedding(config.num_stocks, config.embedding_dim, padding_idx=0)
        static_in = config.num_static_numeric + 2 * config.embedding_dim
        self.static_proj = nn.Linear(static_in, config.hidden_size)
        self.encoder = nn.LSTM(
            input_size=config.num_encoder_features + config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, len(config.quantiles))

    def forward(
        self,
        fund_ids: torch.Tensor,
        stock_ids: torch.Tensor,
        static_numeric: torch.Tensor,
        encoder_numeric: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t, _ = encoder_numeric.shape
        static_vec = torch.cat(
            [self.fund_emb(fund_ids), self.stock_emb(stock_ids), static_numeric], dim=-1
        )
        static_h = self.static_proj(static_vec)                      # (B, H)
        static_seq = static_h.unsqueeze(1).expand(-1, t, -1)         # (B, T, H)
        enc_in = torch.cat([encoder_numeric, static_seq], dim=-1)
        _, (h_n, c_n) = self.encoder(enc_in)
        decoder_in = static_h.unsqueeze(1).expand(-1, self.config.decoder_length, -1)
        dec_out, _ = self.decoder(decoder_in, (h_n, c_n))
        last = dec_out[:, -1, :]
        return {"quantile_pred": self.head(last)}
