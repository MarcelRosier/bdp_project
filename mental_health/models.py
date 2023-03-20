from torch import nn


class SuicideRegressor(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.model(x)


class SuicideRegressorBN(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features=in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=8),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.model(x)
