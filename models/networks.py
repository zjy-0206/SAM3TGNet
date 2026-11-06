import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
from models.SAM3_STENet import SAM3_STENet
import functools
from einops import rearrange

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
#from models.CICNet import CICNet
#from models.p2v import P2VNet
from models.STENet import STENet
from models.MYNet6 import MYNet6
from models.MYNet3D import MYNet3D
from models.Netmodel import AFCF3D_Net
#from models.WNet import WNet


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if isinstance(gpu_ids, int):
        if gpu_ids == -1:
            gpu_ids = []
        else:
            gpu_ids = [gpu_ids]
    if isinstance(gpu_ids, list) and len(gpu_ids) > 0:
        gpu_ids = [int(g) for g in gpu_ids]
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # if args.net_G == 'CrossAttention':
    #     net = Cross_Attention()
    if args.net_G == 'SAM3_STENet':
        net = SAM3_STENet(sam3_checkpoint='E:/STENet/models/sam3_unet/sam3.pt', img_size=args.img_size)

    # elif args.net_G == 'MYNet':
    #     net = MYNet(pretrained=True)
    #
    # elif args.net_G == 'baseline':
    #     net = OnlyTemporal()
    #
    # elif args.net_G == 'P2VNet':
    #     net = P2VNet()

    elif args.net_G == 'STENet':
        net = STENet()

    elif args.net_G == 'MYNet6':
        net = MYNet6()




    elif args.net_G == 'MYNet3D':

        net = MYNet3D(

            sam3_checkpoint=r"E:\STENet\sam3.pt",

            img_size=args.img_size

        )
        if isinstance(gpu_ids, int):
            gpu_ids = [] if gpu_ids == -1 else [gpu_ids]

        if isinstance(gpu_ids, str):
            gpu_ids = [int(x.strip()) for x in gpu_ids.split(',') if int(x.strip()) >= 0]

        if isinstance(gpu_ids, list) and len(gpu_ids) > 0:

            gpu_ids = [int(g) for g in gpu_ids]

            assert torch.cuda.is_available()

            net.to(gpu_ids[0])

            if len(gpu_ids) > 1:
                net = torch.nn.DataParallel(net, gpu_ids)

        return net

    elif args.net_G == 'AFCF3D_Net':
        net = AFCF3D_Net()

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)




if __name__ == '__main__':
    pass
