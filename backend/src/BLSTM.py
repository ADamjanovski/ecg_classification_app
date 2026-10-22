"""BiLSTM ECG classifier."""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    """Two-layer LSTM classifier for PTB-XL multi-label ECG classification."""

    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        feature_size = hidden_size * self.num_directions

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.attn_fc = nn.Linear(feature_size, 1) if use_attention else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        lstm_out, (_, _) = self.lstm(x)

        if self.use_attention:
            if self.attn_fc is None:
                raise RuntimeError("Attention layer is not initialized.")
            attention = torch.softmax(self.attn_fc(lstm_out), dim=1)
            features = (attention * lstm_out).sum(dim=1)
        elif self.bidirectional:
            forward_final = lstm_out[:, -1, : self.hidden_size]
            backward_final = lstm_out[:, 0, self.hidden_size :]
            features = torch.cat([forward_final, backward_final], dim=1)
            attention = None
        else:
            features = lstm_out[:, -1, :]
            attention = None

        logits = self.classifier(features)
        if return_attention:
            if attention is None:
                raise ValueError("return_attention=True requires use_attention=True.")
            return logits, attention.squeeze(-1)
        return logits

    def count_parameters(self) -> int:
        """Return the number of trainable model parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
