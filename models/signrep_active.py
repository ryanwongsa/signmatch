import torch.nn as nn

from models.model_utils.hiera.hiera_nodec_mae import mae_hiera_base_16x224

FEATURE_DIM = 768


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mae_hiera_base_16x224(num_classes=400, pretrained=False)
        self.fc = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, FEATURE_DIM, bias=False)
        )
        self.active_head = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128), nn.GELU(), nn.Linear(128, 2)
        )

    def forward(self, windows):
        latent, _ = self.backbone(windows, mask_ratio=0.0)
        return {"active_logits": self.active_head(self.fc(latent))}
