import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, embed_dim,
                 num_heads: int = 8,
                 attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1. / embed_dim ** 0.5

        self.q = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_heads)])
        self.k = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_heads)])
        self.v = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_heads)])

        # self.qkv = nn.Parameter(torch.randn(1, 8, 3, embed_dim, embed_dim))
        # nn.init.normal_(self.qkv.data)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out = nn.Linear(num_heads * embed_dim, embed_dim)

    def forward(self, x):
        out = []
        for i in range(self.num_heads):
            q = self.q[i](x)
            k = self.k[i](x)
            v = self.v[i](x)

            out.append(self.scale * (q @ k.transpose(-2, -1)).softmax(dim=1) @ v)

        x = torch.concat(out, dim=1)
        x = self.out(x)
        return x


attn = Attention(16)

x = torch.rand(2, 16)

out = attn(x)
print("Done")