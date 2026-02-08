import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self, emb_size, num_classes, s=64.0, m=0.5):
        super().__init__()
        # W: (emb_size, num_classes)
        self.W = nn.Parameter(torch.randn(emb_size, num_classes))
        self.s = s
        self.m = m

    def forward(self, embeddings, labels):
        # исправлено: транспонируем W
        cosine = F.linear(embeddings, F.normalize(self.W).t())
        theta = torch.acos(cosine.clamp(-1+1e-7, 1-1e-7))
        target = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1)

        output = cosine*(1-one_hot) + target*one_hot
        return self.s*output
