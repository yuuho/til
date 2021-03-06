'''
2020/09/29 11:06
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    structure = {
        'decoder'   : [ ['SNTconv4x4', 512, 256], ['InstanceNorm',256],   ['LeakyReLU'],    # 4->8
                        ['SNTconv4x4', 256, 128], ['InstanceNorm',128],   ['LeakyReLU'],    # 8->16
                        ['SNTconv4x4', 128,  64], ['InstanceNorm', 64],   ['LeakyReLU'],    # 16->32
                        ['SNTconv4x4',  64,  32], ['InstanceNorm', 32],   ['LeakyReLU'],    # 32->64
                        ['SNTconv4x4',  32,  16], ['InstanceNorm', 16],   ['LeakyReLU'],    # 64->128
                        ['SNconv1x1' ,  16,   3], ['Tanh']                              ]   # 128
    }

    def _make_block(self,key):
        definition = {
            'SNTconv4x4'      :lambda *config: nn.utils.spectral_norm(nn.ConvTranspose2d(
                                 in_channels=config[0], out_channels=config[1],
                                 kernel_size=4, stride=2, padding=1, bias=True)),
            'SNconv1x1'       :lambda *config: nn.utils.spectral_norm(nn.Conv2d(
                                 in_channels=config[0], out_channels=config[1],
                                 kernel_size=1, stride=1, padding=0, bias=True)),
            'InstanceNorm'  :lambda *config: nn.InstanceNorm2d( config[0],eps=1e-03,affine=True),
            'LeakyReLU'     :lambda *config: nn.LeakyReLU( negative_slope=0.01,inplace=True),
            'Tanh'          :lambda *config: nn.Tanh(),
        }
        return nn.Sequential(*[ definition[k](*cfg) for k,*cfg in self.structure[key]])

    def __init__(self):
        super().__init__()
        self.decoder = self._make_block('decoder')

        def weight_init(m):
            if hasattr(m,'weight'): torch.nn.init.kaiming_normal_(m.weight)
            if hasattr(m,'bias'): torch.nn.init.zeros_(m.bias)
        self.apply(weight_init)


    def forward(self, z):
        # z : (N,512,4,4)
        x = self.decoder(z)
        return x

def Model(*args,**kwargs):
    return nn.DataParallel(Generator(*args,**kwargs))

