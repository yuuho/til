
class Logger:

    def __init__(self,gwriter,writer,savedir,**params):
        super().__init__()
        self.writer = writer
        self.gwriter = gwriter
        self.freq = params['freq']

    def __call__(self, mode, batch_idx, iter_count, epoch_count,
                    loss_g, loss_d, grad_norm, xr, xf,
                    optimG, optimD, generator, discriminator, val_loader, train_loader):

        if mode=='iter_step' and iter_count % self.freq == 0:
            prints = [
                ('loss/all',          {'G'    :loss_g.item(),
                                       'D'    :loss_d.item()+grad_norm.item() },iter_count),
                ('loss/generator',    {'train': loss_g.item()                 },iter_count),
                ('loss/discriminator',{'train': loss_d.item()                 },iter_count),
                ('loss/grad_norm',    {'train': grad_norm.item()              },iter_count)
            ]

            for info in prints:
                self.writer.add_scalars(*info)
                self.gwriter.add_scalars(*info)
