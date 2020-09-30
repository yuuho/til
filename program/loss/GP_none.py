import torch
import torch.nn.functional as F


class Loss(torch.nn.Module):
    def __init__(self,**params):
        super().__init__()
    def forward(self, xr, xf, discriminator, optimD):
        return torch.tensor(0.0,dtype=torch.float32,device=xr.device)

