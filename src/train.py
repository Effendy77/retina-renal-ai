import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, average_precision_score, accuracy_score, f1_score
from .dataset import FundusCSV
from .backbones import create_backbone
from .heads import MLPHead, CoxPHLoss, c_index

def collate_surv(b):
    ims,fs,t=zip(*b)
    d=torch.cat([x[0] for x in t]); e=torch.cat([x[1] for x in t])
    return torch.stack(ims), torch.stack([f for f in fs]), (d,e)

def collate_def(b):
    ims,fs,y=zip(*b); return torch.stack(ims), torch.stack([f for f in fs]), torch.stack(y)

class Model(nn.Module):
    def __init__(self, backbone="resnet50", checkpoint=None, tab_dim=0, task="regression"):
        super().__init__()
        self.bb, dim = create_backbone(backbone, True, checkpoint)
        self.task = task
        self.head = MLPHead(dim+tab_dim, 1, hidden=512)

    def forward(self, x, tab):
        z = self.bb(x)
        if tab is not None and tab.numel()>0: z = torch.cat([z,tab], dim=1)
        return self.head(z)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task", choices=["regression","classification","survival"], required=True)
    ap.add_argument("--train_csv", required=True); ap.add_argument("--val_csv", required=True)
    ap.add_argument("--target", required=True); ap.add_argument("--event_col", default=None)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--backbone", default="resnet50"); ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--outdir", default="outputs/run"); ap.add_argument("--num_workers", type=int, default=2)
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"

    event_col = args.event_col if args.task=="survival" else None
    train = FundusCSV(args.train_csv, args.target, args.task, event_col=event_col, img_size=args.img_size, aug=True)
    val   = FundusCSV(args.val_csv,   args.target, args.task, event_col=event_col, img_size=args.img_size, aug=False)

    tab_dim = train[0][1].numel()
    collate = collate_surv if args.task=="survival" else collate_def
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, collate_fn=collate)
    vl = DataLoader(val,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    model = Model(args.backbone, args.checkpoint, tab_dim, args.task).to(device)
    if args.task=="regression":
        criterion = nn.MSELoss()
    elif args.task=="classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = CoxPHLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = -1e9
    for ep in range(1, args.epochs+1):
        model.train(); tr=[]
        for batch in tl:
            x,t,y = batch; x=x.to(device); t=t.to(device); optim.zero_grad()
            if args.task=="survival":
                d,e = y; d=d.to(device).view(-1); e=e.to(device).view(-1)
                out = model(x,t).view(-1)
                loss = criterion(out, d, e)
            else:
                y = y.to(device)
                out = model(x,t)
                loss = criterion(out, y)
            loss.backward(); optim.step(); tr.append(loss.item())
        print(f"[{ep}] train loss {np.mean(tr):.4f}")

        model.eval()
        with torch.no_grad():
            if args.task=="regression":
                ys, ps = [], []
                for x,t,y in vl:
                    o = model(x.to(device), t.to(device)).cpu().view(-1).numpy()
                    ys.append(y.view(-1).numpy()); ps.append(o)
                import numpy as np
                ys = np.concatenate(ys); ps = np.concatenate(ps)
                from sklearn.metrics import mean_absolute_error, r2_score
                mae = mean_absolute_error(ys, ps); r2 = r2_score(ys, ps)
                metric = -mae
                print(f"  val MAE={mae:.3f} R2={r2:.3f}")
            elif args.task=="classification":
                ys, pr = [], []
                for x,t,y in vl:
                    o = torch.sigmoid(model(x.to(device), t.to(device))).cpu().view(-1).numpy()
                    ys.append(y.view(-1).numpy()); pr.append(o)
                import numpy as np
                ys = np.concatenate(ys); pr = np.concatenate(pr)
                try:
                    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
                    auc = roc_auc_score(ys, pr)
                except Exception:
                    auc = float("nan")
                ap = average_precision_score(ys, pr)
                preds = (pr>=0.5).astype(int)
                acc = accuracy_score(ys, preds); f1 = f1_score(ys, preds, zero_division=0)
                metric = auc if not np.isnan(auc) else ap
                print(f"  val AUC={auc:.3f} AP={ap:.3f} Acc={acc:.3f} F1={f1:.3f}")
            else:
                durs, evts, risks = [], [], []
                for x,t,te in vl:
                    o = model(x.to(device), t.to(device)).cpu().view(-1).numpy()
                    durs.append(te[0].view(-1).numpy()); evts.append(te[1].view(-1).numpy()); risks.append(o)
                import numpy as np
                durs = np.concatenate(durs); evts = np.concatenate(evts); risks = np.concatenate(risks)
                c = c_index(risks, durs, evts); metric = c
                print(f"  val C-index={c:.3f}")

        if metric > best:
            best = metric
            import json
            torch.save(model.state_dict(), os.path.join(args.outdir, "best.pt"))
            with open(os.path.join(args.outdir, "config_snapshot.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            print("  saved best ->", os.path.join(args.outdir, "best.pt"))

if __name__ == "__main__":
    main()
