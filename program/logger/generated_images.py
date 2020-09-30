import numpy as np
import cv2


class Logger:

    def __init__(self,gwriter,writer,savedir,**params):
        super().__init__()
        self.writer = writer
        self.gwriter = gwriter
        assert params['scale'] in {'iter', 'epoch'}
        self.scale = params['scale']
        self.freq = params['freq']
        self.save_freq = params['save_freq']
        self.margin = 3

        self.savedir = savedir / 'generated_images'
        self.savedir.mkdir(parents=True,exist_ok=True)


    def __call__(self, mode, batch_idx, iter_count, epoch_count,
                    loss_g, loss_d, grad_norm, xr, xf,
                    optimG, optimD, generator, discriminator, val_loader, train_loader):

        event_hook = {'iter': 'iter_end', 'epoch': 'epoch_end' }[self.scale]
        event_step = {'iter': iter_count, 'epoch': epoch_count }[self.scale]

        if mode==event_hook:
            
            out = None

            if event_step % self.freq == 0:
            
                # (C,H,W)-Tensor uint8 [0,255]
                out = self.get_output(generator, val_loader)

                self.writer.add_image( 'train/generated',out,iter_count)
                self.gwriter.add_image('train/generated',out,iter_count)

            if event_step % self.save_freq==0:

                image = out.permute(1,2,0).numpy() if out is not None \
                            else self.get_output(generator, val_loader).permute(1,2,0).numpy()
                cv2.imwrite(str(self.savedir/('%08d.png'%iter_count)),image[:,:,::-1])


    def get_output(self, generator, val_loader):
        # 色々と初期化
        device = next(generator.parameters()).device
        generator.eval()

        H = W = 128
        nH = nW = 4
        numImg = nH*nW

        randoms = np.random.RandomState(123)
        constant_noise = randoms.normal(size=(numImg,512,4,4)).astype(np.float32)
        
        batch_size = val_loader.batch_size
        num_calc = (numImg//batch_size + (0 if numImg%batch_size==0 else 1)) # ネットワークに通す回数
        
        _xfs = []
        for i in range(num_calc):
            z = torch.from_numpy(constant_noise[batch_size*i:batch_size*-~i])
            z = z.to(device)
            with torch.no_grad():
                xf = generator(z)
            _xfs += [ ((torch.clamp(xf,-1.0,1.0)+1.0)/2.0*255.0).to(torch.uint8).cpu().clone().detach() ]
        generator.train()
        xfs = torch.cat(_xfs)[:numImg].numpy()
        
        canvas = np.full((3,self.margin*(nH+1)+H*nH, self.margin*(nW+1)+W*nW),255,dtype=np.uint8)
        for i in range(numImg):
            h, w = i//nW, i%nW
            sH = self.margin*(h+1)+H*(h+0)
            sW = self.margin*(w+1)+W*(w+0)
            canvas[:, sH:sH+H, sW:sW+W ] = xfs[i]
        return torch.from_numpy(canvas)

