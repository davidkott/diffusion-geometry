import os
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from load_data import load_CelebAHQ256
from util import training_loss, sampling
from util import rescale, find_max_epoch, print_size

from model import UNet
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
import os



def train(rank, world_size, output_directory, unet_config, ckpt_epoch = -1,  n_epochs=1_000, learning_rate=2e-5,
          batch_size=64,
          T=1_000):
    """
    Train the UNet model on the CELEBA-HQ 256 * 256 dataset

    Parameters:

    output_directory (str):     save model checkpoints to this path
    ckpt_epoch (int or 'max'):  the pretrained model checkpoint to be loaded;
                                automitically selects the maximum epoch if 'max' is selected
    n_epochs (int):             number of epochs to train
    learning_rate (float):      learning rate
    batch_size (int):           batch size
    T (int):                    the number of diffusion steps
    beta_0 and beta_T (float):  diffusion parameters
    unet_config (dict):         dictionary of UNet parameters
    """

    # Compute diffusion hyperparameters
    ddp_setup(rank, world_size)
    steps = T + 1
    s = 0.008
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    Beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    Beta = torch.clip(Beta, 0.0001, 0.9999).to(rank)
    Alpha = 1 - Beta
    Alpha_bar = torch.ones(T).to(rank)
    Beta_tilde = Beta + 0
    Beta=Beta.to(rank)
    Alpha=Alpha.to(rank)
    Alpha_bar=Alpha_bar.to(rank)
    Beta_tilde=Beta_tilde.to(rank)
    for t in range(T):
        Alpha_bar[t] *= Alpha[t] * Alpha_bar[t - 1] if t else Alpha[t]
        if t > 0:
            Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    # Load training data
    trainloader = torch.utils.data.DataLoader(data, batch_size=batch_size // world_size,
                                                             shuffle=False,
                                                             pin_memory=True,
                                                             sampler=torch.utils.data.distributed.DistributedSampler(
                                                                 data),num_workers = 3, drop_last=True)
    print('Data loaded')

    # Predefine model
    net = UNet(**unet_config).to(rank)
    net = DDP(net)
    print_size(net)

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Load checkpoint
    time0 = time.time()
    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(output_directory, 'unet_ckpt')
    if ckpt_epoch >= 0:
        model_path = os.path.join(output_directory, 'unet_ckpt_' + str(ckpt_epoch) + '.pkl')
        checkpoint = torch.load(model_path, map_location='cpu')
        print('Model at epoch %s has been trained for %s seconds' % (ckpt_epoch, checkpoint['training_time_seconds']))
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(rank)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        time0 -= checkpoint['training_time_seconds']
        print('checkpoint model loaded successfully')
    else:
        ckpt_epoch = -1
        print('No valid checkpoint model found, start training from initialization.')

    # Start training
    for epoch in range(ckpt_epoch + 1, n_epochs):
        trainloader.sampler.set_epoch(epoch)
        for i, (X, _) in enumerate(trainloader):
            X = X.to(rank)

            # Back-propagation
            optimizer.zero_grad()
            loss = training_loss(net, nn.MSELoss(), T, X, Alpha_bar,rank)
            loss.backward()
            optimizer.step()

            # Print training loss
            if i % 100 == 0 and rank == 1:
                print("epoch: {}, iter: {}, loss: {:.7f}".format(epoch, i, loss), flush=True)

        # Save checkpoint
        if epoch % 10 == 0:
            if rank == 1:
                torch.save({'epoch': epoch,
                            'model_state_dict': net.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_time_seconds': int(time.time() - time0)},
                           os.path.join(output_directory, 'unet_ckpt_' + str(epoch) + '.pkl'))
                print('model at epoch %s is saved' % epoch)
            else:
                torch.save({'epoch': epoch,
                            'model_state_dict': net.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_time_seconds': int(time.time() - time0)},
                           os.path.join(output_directory, 'IGNORE'))
                print('model at epoch %s is saved' % epoch)


def generate(output_directory, ckpt_path, ckpt_epoch, n,
             T, unet_config):
    """
    Generate images using the pretrained UNet model

    Parameters:

    output_directory (str):     output generated images to this path
    ckpt_path (str):            path of the checkpoints
    ckpt_epoch (int or 'max'):  the pretrained model checkpoint to be loaded; 
                                automitically selects the maximum epoch if 'max' is selected
    n (int):                    number of images to generate
    T (int):                    the number of diffusion steps
    beta_0 and beta_T (float):  diffusion parameters
    unet_config (dict):         dictionary of UNet parameters
    """

    # Compute diffusion hyperparameters
    steps = T + 1
    s = 0.008
    beta_0=.0001
    beta_T = .02
    
    Beta = torch.linspace(beta_0, beta_T, T).cuda()
    Alpha = 1 - Beta
    Alpha_bar = torch.cumprod(Alpha, dim=0)
    Beta_tilde = torch.zeros_like(Beta).cuda()
    for t in range(1, T):
        Beta_tilde[t] = (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t]) * Beta[t]
    Sigma = torch.sqrt(Beta_tilde)

    # Predefine model
    net = UNet( ch=128, out_ch=3, ch_mult=(1,1,2,2,4,4), num_res_blocks=2, attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True, in_channels=3, resolution=256).cuda()
    print_size(net)

    # Load checkpoint
    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(ckpt_path, 'unet_ckpt')
    model_path = os.path.join(ckpt_path, 'unet_ckpt_' + str(ckpt_epoch) + '.pkl')
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print('Model at epoch %s has been trained for %s seconds' % (ckpt_epoch, checkpoint['training_time_seconds']))
        net.load_state_dict(checkpoint['model_state_dict'])
    except:
        raise Exception('No valid model found')

    # Generation
    time0 = time.time()
    X_gen = sampling(net, (n, 3, 256, 256), T, Alpha, Alpha_bar, Sigma)
    print('generated %s samples at epoch %s in %s seconds' % (n, ckpt_epoch, int(time.time() - time0)))

    # Save generated images
    for i in range(n):
        save_image(rescale(X_gen[i]), os.path.join(output_directory, 'img_{}.jpg'.format(i)))
    print('saved generated samples at epoch %s' % ckpt_epoch)


generate(output_directory = '', ckpt_path = r'output/', ckpt_epoch=200, n=5,
             T=1_000,unet_config={})
