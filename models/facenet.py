import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class FaceNet(nn.Module):
    def __init__(self, emb_size=512, num_classes=None):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, emb_size)

        self.classifier = None
        if num_classes:
            self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        emb = self.backbone(x)
        emb = F.normalize(emb)

        if self.classifier:
            return emb, self.classifier(emb)
        return emb
