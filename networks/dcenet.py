import torch.nn as nn
import torch
from utils.utils_torch import sigmoid_normalise, Device

class DCENET(nn.Module):
    def __init__(self, dropout=0.25):
        super(DCENET, self).__init__()
        
        features = 1
        self.rnn = nn.GRU(features, 32, 4, dropout=dropout, batch_first=True, bidirectional=True)
        hidden_dim = 32*2
        self.hidden_dim = hidden_dim

        self.score_ke = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_dt = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_ve = nn.Sequential(nn.Linear(hidden_dim,1), nn.Softmax(dim=1))
        self.score_vp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))

        self.map = nn.Sequential(nn.Linear(80, 1),
                                    nn.ELU())
        self.encoder_ke = nn.Sequential(
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        nn.ELU(),
                                        # nn.BatchNorm1d(int((hidden_dim)/2)),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        
        self.encoder_dt = nn.Sequential(
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        nn.ELU(),
                                        # nn.BatchNorm1d(int((hidden_dim)/2)),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        self.encoder_ve = nn.Sequential(
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        nn.ELU(),
                                        # nn.BatchNorm1d(int((hidden_dim)/2)),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        self.encoder_vp = nn.Sequential(
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        nn.ELU(),
                                        # nn.BatchNorm1d(int((hidden_dim)/2)),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.zeros_(param)
        
    def forward(self, x):
        x = x.unsqueeze(2)
        self.rnn.flatten_parameters()
        output, _ = self.rnn(x)

        score_ke = self.score_ke(output)
        score_dt = self.score_dt(output)
        score_ve = self.score_ve(output)
        score_vp = self.score_vp(output)

        hidden_ke = torch.sum(output*score_ke, dim=1)
        hidden_dt = torch.sum(output*score_dt, dim=1)
        hidden_ve = torch.sum(output*score_ve, dim=1)
        hidden_vp = torch.sum(output*score_vp, dim=1)
        
        
        x_ke = self.encoder_ke(hidden_ke).squeeze(-1)
        x_dt = self.encoder_dt(hidden_dt).squeeze(-1)
        x_ve = self.encoder_ve(hidden_ve).squeeze(-1)
        x_vp = self.encoder_vp(hidden_vp).squeeze(-1)
        
        x = torch.stack([x_ke, x_dt, x_ve, x_vp], dim=1)
        x = sigmoid_normalise(x)

        if torch.isnan(x).any():
            raise ValueError("NaN in output")

        return x


class DCENET_UNC(nn.Module):
    def __init__(self, dropout=0.25):
        super(DCENET_UNC, self).__init__()
        
        features = 1
        self.rnn = nn.GRU(features, 128, 4, dropout=dropout, batch_first=True, bidirectional=True)
        hidden_dim = 256
        self.hidden_dim = hidden_dim

        self.score_ke = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_dt = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_ve = nn.Sequential(nn.Linear(hidden_dim,1), nn.Softmax(dim=1))
        self.score_vp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        self.score_var = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))

        self.map = nn.Sequential(nn.Linear(80, 1),
                                    nn.ELU())
        self.encoder_ke = nn.Sequential(
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        nn.ELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        
        self.encoder_dt = nn.Sequential(
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        nn.ELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        self.encoder_ve = nn.Sequential(
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        nn.ELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        self.encoder_vp = nn.Sequential(
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        nn.ELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int((hidden_dim)/2), 1)
                                        )
        
        self.encoder_var = nn.Sequential(
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, int((hidden_dim)/2)),
                                        nn.ELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(int((hidden_dim)/2), 4)
        )
        
        self.sigmoid = nn.Sigmoid()

        #init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.zeros_(param)
        
    def forward(self, x):
        x = x.unsqueeze(2)
        output, _ = self.rnn(x)


        score_ke = self.score_ke(output)
        score_dt = self.score_dt(output)
        score_ve = self.score_ve(output)
        score_vp = self.score_vp(output)
        score_var = self.score_var(output)

        hidden_ke = torch.sum(output*score_ke, dim=1)
        hidden_dt = torch.sum(output*score_dt, dim=1)
        hidden_ve = torch.sum(output*score_ve, dim=1)
        hidden_vp = torch.sum(output*score_vp, dim=1)
        hidden_var = torch.sum(output*score_var, dim=1)
        
        mu_ke = self.encoder_ke(hidden_ke)
        mu_dt = self.encoder_dt(hidden_dt)
        mu_ve = self.encoder_ve(hidden_ve)
        mu_vp = self.encoder_vp(hidden_vp)
        log_var_ke, log_var_dt, log_var_ve, log_var_vp = torch.chunk(self.encoder_var(hidden_var), 4, dim=-1)

        # Stack the means and standard deviations
        means = torch.stack([mu_ke, mu_dt, mu_ve, mu_vp], dim=1).squeeze(-1)
        log_vars = torch.stack([log_var_ke, log_var_dt, log_var_ve, log_var_vp], dim=1).squeeze(-1)
        if torch.isnan(means).any() or torch.isnan(log_vars).any():
            raise ValueError("Model contains NaN means or standard deviations.")
        if torch.isinf(means).any() or torch.isinf(log_vars).any():
            raise ValueError("Model contains inf means or standard deviations.")

        means = sigmoid_normalise(means)
        out = torch.stack([means, log_vars], dim=1)
        return out
