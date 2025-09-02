import torch, torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class CoxPHLoss(nn.Module):
    def forward(self, log_risk, durations, events):
        order = torch.argsort(durations, descending=True)
        lr = log_risk[order].view(-1); ev = events[order].view(-1)
        denom = torch.logcumsumexp(lr, dim=0)
        return -((lr - denom) * ev).sum() / ev.sum().clamp_min(1.0)

def c_index(risk, durations, events):
    n=len(durations); conc=0; perm=0
    for i in range(n):
        for j in range(n):
            if durations[i]<durations[j] and events[i]==1:
                perm+=1
                if risk[i]>risk[j]: conc+=1
                elif risk[i]==risk[j]: conc+=0.5
    return conc/max(perm,1)
