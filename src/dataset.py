from typing import Optional
import os, cv2, pandas as pd, torch, albumentations as A

SEX_MAP = {'m': 1, 'male': 1, 'f': 0, 'female': 0}

def _norm_sex(x):
    # robustly normalize sex values, tolerate NaN and numeric values
    if pd.isna(x):
        return 1  # default fallback (matches previous behavior)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in SEX_MAP:
            return SEX_MAP[s]
        if s in ('1', 'true', 'yes'):
            return 1
        if s in ('0', 'false', 'no'):
            return 0
        try:
            return int(s)
        except Exception:
            return 1
    try:
        return int(x)
    except Exception:
        return 1

def build_tfms(img_size=224, aug=False):
    if aug:
        return A.Compose([
            A.LongestMaxSize(img_size),
            A.PadIfNeeded(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize()
        ])
    return A.Compose([
        A.LongestMaxSize(img_size),
        A.PadIfNeeded(img_size, img_size),
        A.Normalize()
    ])

class FundusCSV(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, target: str, task: str, event_col: Optional[str] = None, img_size=224, aug=False):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        self.df = pd.read_csv(csv_path)
        self.root = os.path.dirname(csv_path)
        self.target = target
        self.task = task
        self.event_col = event_col
        self.t = build_tfms(img_size, aug)

        # Standardize sex and age
        if 'sex' in self.df:
            self.df['sex_std'] = self.df['sex'].apply(_norm_sex)
        if 'age' in self.df:
            self.df['age_std'] = self.df['age'].astype(float)
        # validate survival inputs
        if self.task == 'survival' and not self.event_col:
            raise ValueError("event_col must be provided for survival task")

    def __len__(self):
        return len(self.df)

    def _read(self, p):
        # Try path as given first; if not absolute, try joining with dataset root
        img = cv2.imread(p)
        if img is None:
            alt = p if os.path.isabs(p) else os.path.join(self.root, p)
            img = cv2.imread(alt)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _feat(self, r):
        # These are the features we want to include if they exist
        cols = [
            'age_std', 'sex_std',
            'egfr', 'acr_mg_g',
            'end_points', 'branch_points',
            'skeleton_length_px', 'mean_width_px',
            'fractal_dim'
        ]
        valid_cols = [c for c in cols if c in self.df.columns]
        if not valid_cols:
            return torch.zeros(0)
        return torch.tensor([float(r[c]) for c in valid_cols], dtype=torch.float32)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # pass the stored image_path to _read; _read will handle relative/absolute paths
        img = self._read(str(r['image_path']))
        img = self.t(image=img)['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        feats = self._feat(r)

        if self.task == 'regression':
            y = torch.tensor([float(r[self.target])], dtype=torch.float32)
            return img, feats, y

        if self.task == 'classification':
            y = torch.tensor([float(r[self.target])], dtype=torch.float32)
            return img, feats, y

        if self.task == 'survival':
            d = torch.tensor([float(r[self.target])], dtype=torch.float32)
            e = torch.tensor([float(r[self.event_col])], dtype=torch.float32)
            return img, feats, (d, e)
        
        if self.task == 'multitask':
            esrd_label = torch.tensor([float(r["esrd_label"])], dtype=torch.float32)
            egfr_value = torch.tensor([float(r["egfr"])], dtype=torch.float32)
            return img, feats, (esrd_label, egfr_value)

        raise ValueError(f"Unsupported task: {self.task}")
