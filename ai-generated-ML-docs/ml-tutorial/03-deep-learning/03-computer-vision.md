# 03 · Computer Vision (CNNs & Transfer Learning)

> **Goal:** Build image models — and know that in practice you **fine-tune pretrained** ones.

## CNN Building Blocks
- **Convolution**: small filters slide over the image, detect edges → textures → objects.
- **Pooling**: downsample, keep strongest signal.
- **Stack**: `Conv → BN → ReLU → Pool` repeated, then a classifier head.
- Famous architectures: LeNet → AlexNet → VGG → **ResNet** (residuals) → EfficientNet → **Vision Transformer (ViT)**.

```python
import torch.nn as nn
class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, n_classes))
    def forward(self, x):          # x: (batch, 3, 32, 32)
        return self.head(self.features(x))
```

## Transfer Learning (what you'll really do)
```python
import torch.nn as nn
from torchvision import models, transforms

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
for p in model.parameters():
    p.requires_grad = False                       # freeze backbone
model.fc = nn.Linear(model.fc.in_features, 5)     # new head for 5 classes

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# Train only model.fc first; then unfreeze everything with a small lr (1e-5) to fine-tune.
```

## CV Task Map
| Task | Output | Go-to tools |
|---|---|---|
| Classification | Label | ResNet, EfficientNet, ViT (`timm`) |
| Object detection | Boxes + labels | YOLO (Ultralytics), DETR |
| Segmentation | Per-pixel mask | U-Net, Mask R-CNN, SAM |
| OCR | Text | Tesseract, PaddleOCR, TrOCR |
| Image generation | Image | Diffusion models |
| Multimodal | Text ↔ image | CLIP, vision-language models |

## Exercises
1. Train `SmallCNN` on CIFAR-10 (target ~75%).
2. Fine-tune ResNet18 on a small custom dataset (e.g., 5 classes of local food) with `ImageFolder`.

---
Next → [Transformers](04-transformers.md)
