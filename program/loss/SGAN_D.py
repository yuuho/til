import torch
import torch.nn.functional as F


class Loss(torch.nn.Module):
    def __init__(self,**params):
        super().__init__()
    def forward(self, fake_judge, real_judge):
        N,C,H,W = fake_judge.shape
        device = fake_judge.device

        loss = F.binary_cross_entropy(fake_judge, torch.zeros(fake_judge.shape,device=device) ) \
                + F.binary_cross_entropy(real_judge, torch.ones(real_judge.shape,device=device) )
        assert loss==( -torch.log(1-fake_judge).mean() -torch.log(real_judge).mean() )
        
        return loss

