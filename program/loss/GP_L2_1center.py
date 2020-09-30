import torch
import torch.nn.functional as F


# one-centered L2
class Loss(torch.nn.Module):
    
    def __init__(self,**params):
        super().__init__()
    
    def forward(self, xr, xf, discriminator, optimD):
        mid = (xr+xf)/2
        mid.requires_grad_()
        judge_mid = discriminator(mid)
        optimD.zero_grad()
        judge_mid.sum().backward(create_graph=True)
        
        # grad は要素毎に勾配が入ったデータ
        # norm(2,dim=1) によってチャンネル方向ベクトルの 2ノルムを計算
        # assert torch.allclose( x.norm(2,dim=1), (x**2).sum(dim=1)**0.5 )

        # (N,C,H,W) -> (N,H,W), 1からの距離を2乗して平均とる -> (1,)
        gradient_penalty = ((mid.grad.norm(2, dim=1) - 1)**2).mean()
        assert torch.allclose( gradient_penalty, (((mid.grad**2).sum(dim=1)**0.5 - 1)**2).mean() )
        return grad_norm

