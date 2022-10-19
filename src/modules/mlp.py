from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GeLU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.mlp(x)
