import torch


class Loss(torch.nn.Module):
    def __init__(self,**params):
        super().__init__()
    def forward(self, fake_judge, real_judge):
        return -fake_judge.mean()
        
