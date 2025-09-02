from typing import Optional
import os, cv2, pandas as pd, torch, albumentations as A

SEX_MAP={'m':1,'male':1,'f':0,'female':0}
def _norm_sex(x):
    if isinstance(x,str):
        x=x.strip().lower()
        return SEX_MAP.get(x, 1 if x in ['1','true','yes'] else 0)
    return int(x)

def build_tfms(img_size=224, aug=False):
    if aug:
        return A.Compose([A.LongestMaxSize(img_size),A.PadIfNeeded(img_size,img_size),A.HorizontalFlip(p=0.5),A.Normalize()])
    return A.Compose([A.LongestMaxSize(img_size),A.PadIfNeeded(img_size,img_size),A.Normalize()])

class FundusCSV(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, target: str, task: str, event_col: Optional[str]=None, img_size=224, aug=False):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        self.df = pd.read_csv(csv_path)
        self.root = os.path.dirname(csv_path)
        self.target = target; self.task = task; self.event_col = event_col
        self.t = build_tfms(img_size, aug)
        if 'sex' in self.df: self.df['sex_std']=self.df['sex'].apply(_norm_sex)
        if 'age' in self.df: self.df['age_std']=self.df['age'].astype(float)
        # Optional: crae, crve, avr

    def __len__(self): return len(self.df)

    def _read(self, p):
        img = cv2.imread(p)
        if img is None: img = cv2.imread(os.path.join(self.root, p))
        if img is None: raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); return img

    def _feat(self, r):
        cols=[c for c in ['age_std','sex_std','crae','crve','avr'] if c in self.df]
        if not cols: return torch.zeros(0)
        return torch.tensor([float(r[c]) for c in cols], dtype=torch.float32)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self._read(os.path.join(self.root, str(r['image_path'])))
        img = self.t(image=img)['image']
        img = torch.from_numpy(img).permute(2,0,1).float()
        feats = self._feat(r)
        if self.task=='regression':
            y=torch.tensor([float(r[self.target])],dtype=torch.float32); return img,feats,y
        if self.task=='classification':
            y=torch.tensor([float(r[self.target])],dtype=torch.float32); return img,feats,y
        if self.task=='survival':
            d=torch.tensor([float(r[self.target])],dtype=torch.float32)
            e=torch.tensor([float(r[self.event_col])],dtype=torch.float32)
            return img,feats,(d,e)
        raise ValueError(self.task)
