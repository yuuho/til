'''
2020/09/29 10:45

'''
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from hellflame.snippets.torch_gpu import _set_device
from hellflame.snippets.config_loader import _load_modules


class Trainer:

    def __init__(self,config,**params):
        self.config = config

        print('devices setting...')
        self._set_device()
        print('modules loading...')
        self._load_modules()
        print('modules constructing...')
        self._construct_models()

        if self.config['env']['is_continue']:
            self._continue_models()
        

    # スニペット
    _set_device = _set_device
    _load_modules = _load_modules


    def _construct_models(self):
        mods, cfg = self.modules, self.config

        # データセットの読み込み
        self.train_dataset = mods.DatasetTrain(cfg['env']['data'],cfg['env']['tmp'],**cfg['dataset']['train']['params'])
        self.val_dataset = mods.DatasetVal(cfg['env']['data'],cfg['env']['tmp'],**cfg['dataset']['val']['params'])
        
        # # データローダーの作成
        self.train_loader = mods.LoaderTrain(self.train_dataset, **cfg['loader']['train']['params'])
        self.val_loader = mods.LoaderVal(self.val_dataset, **cfg['loader']['val']['params'])

        # モデルの作成
        self.generator = mods.ModelGenerator(**cfg['model']['generator']['params']).to(self.device)
        self.discriminator = mods.ModelDiscriminator(**cfg['model']['discriminator']['params']).to(self.device)

        # 最適化手法の設定
        self.optimizerG = mods.OptimizerGenerator(self.generator, **cfg['optimizer']['generator']['params'])
        self.optimizerD = mods.OptimizerDiscriminator(self.discriminator, **cfg['optimizer']['discriminator']['params'])
        
        # 誤差関数の設定
        self.lossG  = mods.LossGenerator(**cfg['loss']['generator']['params']).to(self.device)
        self.lossD  = mods.LossDiscriminator(**cfg['loss']['discriminator']['params']).to(self.device)
        self.lossGP = mods.LossGradientpenalty(**cfg['loss']['gradientpenalty']['params']).to(self.device)

        # ロガーの設定
        self.writer = SummaryWriter(cfg['env']['savedir']/'logdir')
        self.global_writer = SummaryWriter(cfg['env']['tmp']/'logdir'/(cfg['env']['exp_name']).replace('/','-'))
        self.loggers = [ m(self.global_writer,self.writer,cfg['env']['savedir'], **d['params'])
                                    for m,d in zip(mods.Loggers,cfg['logger'])]

    def _continue_models(self):
        print('implemented error')
        assert False

    def train(self):
        # 使用するものを名前空間へ展開
        device = self.device
        train_loader, val_loader = self.train_loader, self.val_loader
        generator, discriminator = self.generator, self.discriminator
        optimG, optimD = self.optimizerG, self.optimizerD
        lossfuncG, lossfuncD, lossfuncGP = self.lossG, self.lossD, self.lossGP

        log_vals = ['batch_idx', 'iter_count', 'epoch_count',
                    'loss_g', 'loss_d', 'grad_norm', 'xr', 'xf',
                    'optimG', 'optimD', 'generator', 'discriminator', 'val_loader', 'train_loader']
        log_getv = lambda ns: [ ns[k] if k in ns.keys() else None  for k in log_vals]
        log = lambda mode,ns: [l(mode,*log_getv(ns)) for l in self.loggers]
        #log = lambda mode,ns: ['nothing']

        # 学習ループ
        iter_count = epoch_count = 0
        requires_new_batchsize = True
        end_flag = False

        while True:

            # エポックの初期化処理
            if 'end' in log('epoch_start',locals()): end_flag=True; break

            for batch_idx, xr in enumerate(train_loader):

                # イテレーションの初期化処理
                if 'end' in log('iter_start',locals()): end_flag=True; break

                N,C,H,W = xr.shape

                # デバイスへ転送
                z = torch.empty((N,512,4,4),dtype=torch.float32,device=device)
                torch.nn.init.normal_(z)
                xr = xr.to(device)
                
                # 識別器の学習
                with torch.no_grad(): xf = generator(z).detach()
                judge_xf = discriminator(xf)
                judge_xr = discriminator(xr)
                loss_d = lossfuncD(judge_xr, judge_xf)
                grad_norm = lossfuncGP(xr,xf, discriminator, optimD)
                loss_d_all = loss_d + grad_norm

                optimD.zero_grad()
                loss_d_all.backward()
                optimD.step()
                
                # 生成器の学習
                xf = generator(z)
                judge_xf = discriminator(xf)
                judge_xr = discriminator(xr)
                loss_g = lossfuncG(judge_xr, judge_xf)

                optimG.zero_grad()
                loss_g.backward()
                optimG.step()

                # イテレーションの終了処理
                iter_count += 1
                if 'end' in log('iter_step',locals()): end_flag=True; break
                del loss_g, loss_d, grad_norm, loss_d_all, z, xr, xf
                if 'end' in log('iter_end',locals()): end_flag=True; break

            # エポックの終了処理
            del batch_idx
            epoch_count += 1
            if end_flag: break
            if 'end' in log('epoch_end',locals()): break

        self.writer.close()
        self.global_writer.close()
        return True

