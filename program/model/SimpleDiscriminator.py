
import torch
import torch.nn as nn


class SimpleDiscriminator(nn.Module):
    structure = {
        'encoder'   : [ ['SNconv1x1',   3,  16], ['InstanceNorm', 16],   ['LeakyReLU'],    # 128->128
                        ['SNconv3x3',  16,  32], ['InstanceNorm', 32],   ['LeakyReLU'],    # 128->64
                        ['SNconv3x3',  32,  64], ['InstanceNorm', 64],   ['LeakyReLU'],    # 64->32
                        ['SNconv3x3',  64, 128], ['InstanceNorm',128],   ['LeakyReLU'],    # 32->16
                        ['SNconv3x3', 128, 256], ['InstanceNorm',256],   ['LeakyReLU'],    # 16->8
                        ['SNconv3x3', 256, 512], ['InstanceNorm',512],   ['LeakyReLU'],    # 8->4
                        ['SNconv3x3', 512, 512], ['InstanceNorm',512],   ['LeakyReLU'],    # 4->2
                        ['SNconv3x3', 512, 512]                                           ]# 2->1
    }

    def _make_block(self,key):
        definition = {
            'SNconv3x3'     :lambda *config: nn.utils.spectral_norm(nn.Conv2d(
                                 in_channels=config[0], out_channels=config[1],
                                 kernel_size=3, stride=2, padding=1, bias=True)),
            'SNconv1x1'     :lambda *config: nn.utils.spectral_norm(nn.Conv2d(
                                 in_channels=config[0], out_channels=config[1],
                                 kernel_size=1, stride=1, padding=0, bias=True)),
            'InstanceNorm'  :lambda *config: nn.InstanceNorm2d( config[0],eps=1e-03,affine=True),
            'LeakyReLU'     :lambda *config: nn.LeakyReLU( negative_slope=0.01,inplace=True),
            'Sigmoid'       :lambda *config: nn.Sigmoid(),
        }
        return nn.Sequential(*[ definition[k](*cfg) for k,*cfg in self.structure[key]])

    def __init__(self):
        super().__init__()
        self.encoder = self._make_block('encoder')

        def weight_init(m):
            if hasattr(m,'weight'): torch.nn.init.kaiming_normal_(m.weight)
            if hasattr(m,'bias'): torch.nn.init.zeros_(m.bias)
        self.apply(weight_init)

    def forward(self, x):
        # x : (N,3,128,128)
        y = self.encoder( x )
        return y

def Model(*args,**kwargs):
    return nn.DataParallel(SimpleDiscriminator(*args,**kwargs))

