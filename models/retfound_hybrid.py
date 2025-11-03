
import torch
import torch.nn as nn
from transformers import AutoModel

class RetFoundHybrid(nn.Module):
    def __init__(self, retfound_model_path: str, tabular_feat_dim: int, dropout: float = 0.2):
        super().__init__()
        self.retfound = AutoModel.from_pretrained(retfound_model_path, trust_remote_code=True)

        image_feat_dim = self.retfound.config.hidden_size  # usually 768 or 1024
        fusion_input_dim = image_feat_dim + tabular_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, image_tensor, tabular_tensor):
        retfound_output = self.retfound(pixel_values=image_tensor)
        img_embedding = retfound_output.pooler_output
        fused = torch.cat((img_embedding, tabular_tensor), dim=1)
        logits = self.classifier(fused)
        return logits
