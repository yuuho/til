import os

import tqdm
import torch


class Logger:

    def __init__(self,gwriter,writer,savedir,**kwargs):
        self.writer = writer
        self.gwriter = gwriter

    def __call__(self, mode, batch_idx, iter_count, epoch_count,
                    loss_g, loss_d, grad_norm, xr, xf,
                    optimG, optimD, generator, discriminator, val_loader, train_loader):
        
        if mode=='epoch_start':
            print("   See in 'tensorboard --logdir %s' or %s"%(str(self.writer.log_dir),str(self.gwriter.log_dir.parent)))
            self.pbar = tqdm.tqdm(total=len(train_loader))
            self.pbar.set_description("%s [Epoch %4d]"%(os.environ.get('CUDA_VISIBLE_DEVICES'),epoch_count+1))

        if mode=='iter_step':
            self.pbar.update(1)

        if mode=='epoch_end':
            self.pbar.close()
