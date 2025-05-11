import torch
import torch.nn as nn
from utils.utils_torch import sigmoid_normalise
from utils.utils_torch import Device

class FC(nn.Module):
    def __init__(self, input_size=80, hidden_size=100, output_size=4, dropout=0.25):
        super(FC, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size*2),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(Device)

        # self.sigmoid = nn.Sigmoid()

        self.init_weights()


    def forward(self, x):
        self.model.to(Device)
        out = self.model(x)
        out = sigmoid_normalise(out)
        if torch.isnan(out).any():
            print("Warning: NaN detected in output")
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class FC_UNC(nn.Module):
    def __init__(self, input_size=80, hidden_size=100, output_size=4, dropout=0.25):
        super(FC_UNC, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size*2),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size*2)
        )
        self.sigmoid = nn.Sigmoid()

        self.num_outputs = output_size

        self.init_weights()

    def forward(self, x):
        self.model.to(Device)
        mean, log_var = torch.chunk(self.model(x), 2, dim=-1)
        mean = sigmoid_normalise(mean)
        if torch.isnan(mean).any():
            print("Warning: NaN detected in output")

        out = torch.stack([mean, log_var], dim=1).squeeze(-1)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


