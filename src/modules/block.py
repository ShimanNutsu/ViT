from torch import nn


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.norm_pre = nn.LayerNorm(dim)
        self.attention = Attention()
        self.mlp = MLP()
        self.norm_pos = nn.LayerNorm(dim)

    def forward(self, x):
        xq = self.norm_pre(x)
        xq = self.attention()
        x = xq + x
        xn = self.norm_pos(x)
        xn = self.mlp(xn)
        return x + xn