
# In your train.py

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class HybridDataset(Dataset):
    def __init__(self, csv_file, image_dir, tabular_features, target_column, image_size):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tabular_features = tabular_features
        self.target_column = target_column
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.image_dir}/{row['filename']}"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tabular = torch.tensor(row[self.tabular_features].values.astype(float), dtype=torch.float)
        label = torch.tensor(row[self.target_column], dtype=torch.float)
        return image, tabular, label
