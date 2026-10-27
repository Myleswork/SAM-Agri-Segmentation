import torch
from torch import nn

class SpectralStem(nn.Module):
    """Project normalized channels to normalized RGB; RGB identity initially."""
    def __init__(self, channels):
        super().__init__()
        if len(set(channels)) != len(channels): raise ValueError('channel names must be unique')
        try: rgb=[channels.index(c) for c in ('rgb_r','rgb_g','rgb_b')]
        except ValueError as exc: raise ValueError('RGB channels required') from exc
        self.projection=nn.Conv2d(len(channels),3,1,bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            for i,j in enumerate(rgb): self.projection.weight[i,j,0,0]=1
    def forward(self,x): return self.projection(x)
