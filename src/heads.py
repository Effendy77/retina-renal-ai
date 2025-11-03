import torch, torch.nn as nn
import numpy as np

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class MultiTaskHead(nn.Module):
    """
    For hybrid multi-task learning, e.g. predict ESRD and eGFR simultaneously.
    Each head is a separate MLP.
    """
    def __init__(self, in_dim, out_dims, hidden=512, drop=0.2):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
                nn.Linear(hidden, out_dim)
            ) for out_dim in out_dims
        ])

    def forward(self, x):
        return [head(x) for head in self.heads]


class CoxPHLoss(nn.Module):
    def forward(self, log_risk, durations, events):
        order = torch.argsort(durations, descending=True)
        lr = log_risk[order].view(-1)
        ev = events[order].view(-1)
        denom = torch.logcumsumexp(lr, dim=0)
        return -((lr - denom) * ev).sum() / ev.sum().clamp_min(1.0)


def c_index(risk, durations, events):
    # support torch tensors or numpy arrays / lists
    if hasattr(risk, "detach"):
        risk = risk.detach().cpu().numpy().ravel()
    else:
        risk = np.asarray(risk).ravel()
    if hasattr(durations, "detach"):
        durations = durations.detach().cpu().numpy().ravel()
    else:
        durations = np.asarray(durations).ravel()
    if hasattr(events, "detach"):
        events = events.detach().cpu().numpy().ravel()
    else:
        events = np.asarray(events).ravel()

    n = len(durations)
    conc = 0.0
    perm = 0
    for i in range(n):
        for j in range(n):
            if durations[i] < durations[j] and events[i] == 1:
                perm += 1
                if risk[i] > risk[j]:
                    conc += 1.0
                elif risk[i] == risk[j]:
                    conc += 0.5
    return conc / max(perm, 1)
