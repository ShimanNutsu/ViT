import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.norm_pre = nn.LayerNorm(dim)
        self.attention = Attention(dim, 64, num_heads, drop_rate, drop_rate)
        self.mlp = MLP(dim, dim * mlp_ratio, dim, drop_rate)
        self.norm_pos = nn.LayerNorm(dim)

    def forward(self, x):
        xq = self.norm_pre(x)
        xq = self.attention(xq)
        x = xq + x
        xn = self.norm_pos(x)
        xn = self.mlp(xn)
        return x + xn


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1. / hidden_dim ** 0.5

        self.q = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_heads)])
        self.k = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_heads)])
        self.v = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_heads)])

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Linear(num_heads * hidden_dim, input_dim)

    def forward(self, x):
        q = [self.proj_dropout(proj(x)) for proj in self.q]
        k = [self.proj_dropout(proj(x)) for proj in self.k]
        v = [self.proj_dropout(proj(x)) for proj in self.v]
        heads = [self.softmax(self.scale *
                              q[i].bmm(torch.transpose(k[i], 1, 2))).bmm(v[i])
                 for i in range(self.num_heads)]
        x = torch.cat(heads, dim=2)
        x = self.attn_dropout(self.out(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        x = self.act(self.lin1(x))
        x = self.lin2(x)
        return x
