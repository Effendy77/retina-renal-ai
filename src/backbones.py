import torch
import timm
from transformers import AutoModel
from torchvision.models import resnet18


def create_backbone(name="resnet50", pretrained=True, checkpoint=None):
    if name.lower() == "retfound":
        # Load RetFound from Hugging Face using its model path
        hf_id = checkpoint or name
        model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)

        # Try to find the feature dimension
        feat_dim = getattr(model.config, "hidden_size", None)
        if feat_dim is None:
            feat_dim = getattr(model.config, "vision_embed_dim", None)
        if feat_dim is None:
            feat_dim = getattr(model.config, "projection_dim", None)
        if feat_dim is None:
            feat_dim = 2048  # fallback
        return RetFoundWrapper(model), feat_dim

    elif name.lower() == "deepdkd":
        model = resnet18(pretrained=False)
        feat_dim = model.fc.in_features
        model.fc = torch.nn.Identity()  # remove final classification head

        try:
            sd = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(sd, strict=False)
            print("[DeepDKD] checkpoint loaded.")
        except Exception as e:
            print(f"[DeepDKD] failed to load weights: {e}")

        return model, feat_dim

    # Default: timm CNN models (e.g., resnet50, efficientnet, etc.)
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    feat_dim = getattr(m, "num_features", 2048)

    if checkpoint:
        try:
            sd = torch.load(checkpoint, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            msd = m.state_dict()
            clean = {}
            for k, v in sd.items():
                k2 = k.replace("module.", "")
                if k2 in msd and msd[k2].shape == v.shape:
                    clean[k2] = v
            m.load_state_dict(clean, strict=False)
            print("[backbone] checkpoint loaded.")
        except Exception as e:
            print(f"[backbone] failed to load checkpoint: {e}")

    return m, feat_dim


class RetFoundWrapper(torch.nn.Module):
    """
    Wraps Hugging Face RetFound model to return pooled image embeddings,
    compatible with the hybrid training pipeline.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(pixel_values=x)
        # Prefer attribute-style outputs
        if hasattr(out, "pooler_output") and getattr(out, "pooler_output") is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state.mean(dim=1)
        # Dict-style fallback
        if isinstance(out, dict):
            if "pooler_output" in out and out["pooler_output"] is not None:
                return out["pooler_output"]
            if "last_hidden_state" in out:
                return out["last_hidden_state"].mean(dim=1)
        # Model helper fallback
        if hasattr(self.model, "get_image_features"):
            return self.model.get_image_features(pixel_values=x)
        raise RuntimeError("Unable to extract pooled embeddings from RetFound model output")
