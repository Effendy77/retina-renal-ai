import torch
import torch.nn as nn
from .backbones import create_backbone
from .heads import MultiTaskHead

class MultiTaskModel(nn.Module):
    def __init__(self, backbone="resnet50", checkpoint=None, tab_dim=0):
        super().__init__()
        self.bb, dim = create_backbone(backbone, True, checkpoint)
        self.head = MultiTaskHead(dim + tab_dim, out_dims=[1, 1])  # [ESRD, EGFR]

    def forward(self, x, tab=None):
        """
        Args:
            x: Image tensor [B, C, H, W]
            tab: Tabular features [B, tab_dim] or None
        Returns:
            [esrd_logits, egfr_pred]: Multitask predictions
        """
        z = self.bb(x)
        if tab is not None and tab.numel() > 0:
            z = torch.cat([z, tab], dim=1)
        return self.head(z)  # returns [esrd_logits, egfr_pred]
