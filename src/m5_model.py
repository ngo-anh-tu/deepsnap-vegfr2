"""
M5 - Backbone timm (ResNet-18 / EfficientNet-B0) + dau hoi quy 1 chieu.

  forward(x[B,V,3,H,W]) -> du bao TUNG ANH [B,V]
  aggregate(pred[B,V])  -> du bao muc phan tu [B], la median qua V (quantile 0.5).
Khong re nhanh theo V: V=1 thi median cua mot phan tu la chinh no (§12.8 #2); V27 dung lai ham nay.
Nhan duoc chuan hoa theo train (y_mean, y_std luu trong buffer) nen dau ra la pIC50 that.
"""
import timm, torch, torch.nn as nn


class MolRegressor(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, drop_rate: float, y_mean: float, y_std: float):
        super().__init__()
        self.body = timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        self.head = nn.Linear(self.body.num_features, 1)
        pc = getattr(self.body, 'pretrained_cfg', {}) or {}
        mean, std = pc.get('mean', (0.485, 0.456, 0.406)), pc.get('std', (0.229, 0.224, 0.225))
        self.register_buffer('img_mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor(std).view(1, 3, 1, 1))
        self.register_buffer('y_mean', torch.tensor(float(y_mean)))
        self.register_buffer('y_std', torch.tensor(float(y_std)))

    def predict_images(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N,3,H,W] trong [0,1] -> [N] pIC50."""
        z = self.head(self.body((x - self.img_mean) / self.img_std)).squeeze(-1)
        return z * self.y_std + self.y_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v = x.shape[:2]
        return self.predict_images(x.flatten(0, 1)).view(b, v)


def aggregate(pred_bv: torch.Tensor) -> torch.Tensor:
    return torch.quantile(pred_bv.float(), 0.5, dim=1)


def build_model(cfg: dict, y_mean: float, y_std: float) -> MolRegressor:
    m = cfg['model']
    return MolRegressor(m['backbone'], m['pretrained'], m['drop_rate'], y_mean, y_std)
