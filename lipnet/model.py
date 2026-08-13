"""LipNet model, minimally adapted from VIPL's ``model.py``.

Upstream source:
https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch/blob/40209e09c49553c00c25c7d41faa3706aea3c625/model.py

The architecture and forward pass are intentionally kept recognizable.  The
only functional change is the ``num_classes`` constructor argument: 28 keeps
the original GRID model, while the Serbian adapter uses 29 CTC classes.
See ``LICENSE.vipl`` and ``docs/upstream-diff.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.init as init


class LipNet(torch.nn.Module):
    """VIPL 3D-CNN + two BiGRU layers + linear CTC head."""

    def __init__(self, dropout_p: float = 0.5, num_classes: int = 28):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes mora da uključi CTC blank i bar jedan simbol")

        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512, num_classes)
        self.dropout_p = dropout_p
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self._init()

    @property
    def num_classes(self) -> int:
        return self.FC.out_features

    def _init(self) -> None:
        # Initialization is copied from VIPL model.py, including its GRU rule.
        init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        init.constant_(self.conv1.bias, 0)
        init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        init.constant_(self.conv2.bias, 0)
        init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        init.constant_(self.conv3.bias, 0)
        init.kaiming_normal_(self.FC.weight, nonlinearity="sigmoid")
        init.constant_(self.FC.bias, 0)

        for recurrent_layer in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for offset in range(0, 256 * 3, 256):
                init.uniform_(
                    recurrent_layer.weight_ih_l0[offset : offset + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(recurrent_layer.weight_hh_l0[offset : offset + 256])
                init.constant_(recurrent_layer.bias_ih_l0[offset : offset + 256], 0)
                init.uniform_(
                    recurrent_layer.weight_ih_l0_reverse[offset : offset + 256],
                    -math.sqrt(3) * stdv,
                    math.sqrt(3) * stdv,
                )
                init.orthogonal_(
                    recurrent_layer.weight_hh_l0_reverse[offset : offset + 256]
                )
                init.constant_(
                    recurrent_layer.bias_ih_l0_reverse[offset : offset + 256], 0
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Očekivan (B,C,T,H,W) ulaz, dobijeno {tuple(x.shape)}")
        if x.shape[1] != 3 or x.shape[-2:] != (64, 128):
            raise ValueError(
                "VIPL LipNet očekuje 3 kanala i spatialni oblik 64x128; "
                f"dobijeno {tuple(x.shape)}"
            )

        x = self.pool1(self.dropout3d(self.relu(self.conv1(x))))
        x = self.pool2(self.dropout3d(self.relu(self.conv2(x))))
        x = self.pool3(self.dropout3d(self.relu(self.conv3(x))))

        # (B,C,T,H,W) -> (T,B,C,H,W) -> (T,B,C*H*W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x)

        x = self.FC(x)
        return x.permute(1, 0, 2).contiguous()
