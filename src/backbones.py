import torch, timm

def create_backbone(name="resnet50", pretrained=True, checkpoint=None):
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    feat_dim = getattr(m, "num_features", 2048)
    if checkpoint:
        try:
            sd = torch.load(checkpoint, map_location="cpu")
            if "state_dict" in sd: sd = sd["state_dict"]
            msd = m.state_dict()
            clean = {}
            for k,v in sd.items():
                k2 = k.replace("module.","")
                if k2 in msd and msd[k2].shape == v.shape:
                    clean[k2] = v
            m.load_state_dict(clean, strict=False)
            print("[backbone] checkpoint loaded.")
        except Exception as e:
            print(f"[backbone] failed to load checkpoint: {e}")
    return m, feat_dim
