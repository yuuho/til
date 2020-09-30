import torch
import torch.nn.functional as F


class Loss(torch.nn.Module):
    def __init__(self,**params):
        super().__init__()
    def forward(self, fake_judge, real_judge):
        return -real_judge.mean() + fake_judge.mean()
