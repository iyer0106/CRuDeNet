import torch
from torch import nn


class ConvGRUCell(nn.Module):
    def __init__(self, c_in: int, c_hid: int, k: int = 3) -> None:
        super().__init__()
        self.c_hid = c_hid
        pad = k // 2
        self.convz = nn.Conv2d(c_in + c_hid, c_hid, k, padding=pad)
        self.convr = nn.Conv2d(c_in + c_hid, c_hid, k, padding=pad)
        self.convq = nn.Conv2d(c_in + c_hid, c_hid, k, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.size(0), self.c_hid, x.size(2), x.size(3), device=x.device)
        inp = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.convz(inp))
        r = torch.sigmoid(self.convr(inp))
        q = torch.tanh(self.convq(torch.cat([x, r * h], dim=1)))
        h = (1 - z) * h + z * q
        return h


class ConvGRU(nn.Module):
    """Causal temporal model (forward-only) operating per-frame.

    Expects input shape [B, 1, T, H, W] and returns same shape.
    """

    def __init__(self, c_in: int = 1, c_hid: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ConvGRUCell(c_in if i == 0 else c_hid, c_hid)
            for i in range(num_layers)
        ])
        self.out_conv = nn.Conv2d(c_hid, 1, 3, padding=1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, 1, T, H, W]
        B, _, T, H, W = x_seq.shape
        h_states = [None] * len(self.layers)
        outs = []
        for t in range(T):
            x_t = x_seq[:, :, t]
            for i, cell in enumerate(self.layers):
                h_states[i] = cell(x_t, h_states[i])
                x_t = h_states[i]
            outs.append(self.out_conv(x_t))
        return torch.stack(outs, 2)


def build_model(c_in: int = 1, c_hid: int = 32, num_layers: int = 3) -> ConvGRU:
    return ConvGRU(c_in=c_in, c_hid=c_hid, num_layers=num_layers)


