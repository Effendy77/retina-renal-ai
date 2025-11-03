import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, average_precision_score, accuracy_score, f1_score

from .dataset import FundusCSV
from .model_multitask import MultiTaskModel

def collate_def(batch):
    ims, feats, targets = zip(*batch)
    ims = torch.stack(ims)
    # feats can be a list of zero-dimensional tensors or empty tensors
    if feats[0].numel() > 0:
        feats = torch.stack(feats)
    else:
        feats = torch.zeros(len(batch), 0)
    # targets should be stacked into shape [B, 2]
    targets = torch.stack([torch.tensor([t[0], t[1]]) for t in targets])
    return ims, feats, targets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--outdir", default="outputs/run_multitask")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--task", choices=["multitask"], default="multitask", help="Task type")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = FundusCSV(args.train_csv, target="dummy", task="multitask", img_size=args.img_size, aug=True)
    val = FundusCSV(args.val_csv, target="dummy", task="multitask", img_size=args.img_size, aug=False)

    sample = train[0]
    tab_dim = sample[1].numel() if sample[1] is not None else 0

    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_def)
    vl = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_def)

    model = MultiTaskModel(args.backbone, args.checkpoint, tab_dim).to(device)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        tr = []
        for x, t, y in tl:
            x, t, y = x.to(device), t.to(device), y.to(device)
            esrd_target = y[:, 0]
            egfr_target = y[:, 1]
            optim.zero_grad()
            esrd_out, egfr_out = model(x, t)
            loss_cls = criterion_cls(esrd_out.view(-1), esrd_target)
            loss_reg = criterion_reg(egfr_out.view(-1), egfr_target)
            loss = loss_cls + loss_reg
            loss.backward()
            optim.step()
            tr.append(loss.item())
        print(f"[{ep}] train loss {np.mean(tr):.4f}")

        model.eval()
        esrd_true, esrd_pred = [], []
        egfr_true, egfr_pred = [], []
        with torch.no_grad():
            for x, t, y in vl:
                x, t, y = x.to(device), t.to(device), y.to(device)
                o1, o2 = model(x, t)
                esrd_true.append(y[:, 0].cpu().numpy())
                esrd_pred.append(torch.sigmoid(o1).cpu().view(-1).numpy())
                egfr_true.append(y[:, 1].cpu().numpy())
                egfr_pred.append(o2.cpu().view(-1).numpy())

        esrd_true = np.concatenate(esrd_true)
        esrd_pred = np.concatenate(esrd_pred)
        egfr_true = np.concatenate(egfr_true)
        egfr_pred = np.concatenate(egfr_pred)

        try:
            auc = roc_auc_score(esrd_true, esrd_pred)
        except:
            auc = float("nan")
        avg_prec = average_precision_score(esrd_true, esrd_pred)
        acc = accuracy_score(esrd_true, esrd_pred >= 0.5)
        f1 = f1_score(esrd_true, esrd_pred >= 0.5, zero_division=0)
        mae = mean_absolute_error(egfr_true, egfr_pred)
        r2 = r2_score(egfr_true, egfr_pred)
        metric = auc if not np.isnan(auc) else avg_prec

        print(f"  val ESRD AUC={auc:.3f}, AP={avg_prec:.3f}, Acc={acc:.3f}, F1={f1:.3f}")
        print(f"  val eGFR MAE={mae:.3f}, R2={r2:.3f}")

        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), os.path.join(args.outdir, "best.pt"))
            print("  saved best model")

if __name__ == "__main__":
    main()