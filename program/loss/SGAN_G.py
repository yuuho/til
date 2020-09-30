import torch
import torch.nn.functional as F


class Loss(torch.nn.Module):
    def __init__(self,**params):
        super().__init__()
    def forward(self, _, judge_xf):
        N,C,H,W = judge_xf.shape
        device = judge_xf.device
        dtype = judge_xf.dtype

        loss = F.binary_cross_entropy(judge_xf, torch.ones(judge_xf.shape, dtype=dtype,device=device) )
        assert loss==-torch.log(judge_xf).mean()
        
        return loss

