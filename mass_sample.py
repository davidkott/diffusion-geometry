import os
import argparse
import json
import time
import multiprocessing
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from util import training_loss,std_normal
from util import rescale, find_max_epoch, print_size
from model import UNet
import torch.distributed as dist
import os
import time


EPSILON = 1
IMAGES_PER_GPU = 40


def fixed_sampling(net, size, T, Alpha, Alpha_bar, Sigma,rank,START):
    """
    Perform the complete sampling step according to p(x_0|x_T)
    """
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 4
    #print('begin sampling, total steps = %s' % T)
    
    # NOTE: changing the coeffecient will increase the variance of pictures, at .3 they all look similar but at 1.0 I got a woman and a bald man
    x = torch.zeros(size,device=rank).float() + torch.load('sample_at_'+str(START)+'.pt',map_location='cpu').to(rank)
    
    with torch.no_grad():
        for t in range(START,-1,-1):
            #if t % 100 == 0:
            #    print('reverse step:', t)
            ts = (t * torch.ones(size[0])).to(rank)
            epsilon_theta = net(x,ts)
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            
            if t > 0:
                x = x + (Sigma[t] * torch.randn(size,device=rank).float() )
    return x

def generate(output_directory, ckpt_path, ckpt_epoch, n,
             T, beta_0, beta_T, unet_config,rank,START):
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
    #Beta = torch.linspace(beta_0, beta_T, T).to(rank)
    #Alpha = 1 - Beta
    #Alpha_bar = torch.ones(T).to(rank)
    #Beta_tilde = Beta + 0
    #for t in range(T):
    #    Alpha_bar[t] *= Alpha[t] * Alpha_bar[t - 1] if t else Alpha[t]
    #    if t > 0:
    #        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
    #Sigma = torch.sqrt(Beta_tilde)
    
    Beta = torch.linspace(beta_0, beta_T, T).to(rank)
    Alpha = 1 - Beta
    Alpha_bar = torch.cumprod(Alpha, dim=0)
    Beta_tilde = torch.zeros_like(Beta)
    for t in range(1, T):
        Beta_tilde[t] = (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t]) * Beta[t]
    Sigma = torch.sqrt(Beta_tilde)

    # Predefine model
    net = UNet( ch=128, out_ch=3, ch_mult=(1,1,2,2,4,4), num_res_blocks=2, attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True, in_channels=3, resolution=256).to(rank)
    #print_size(net)

    # Load checkpoint
    if ckpt_epoch == 'max':
        ckpt_epoch = find_max_epoch(ckpt_path, 'unet_ckpt')
    model_path = os.path.join(ckpt_path, 'unet_ckpt_' + str(ckpt_epoch) + '.pkl')
    print(model_path)
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        #print('Model at epoch %s has been trained for %s seconds' % (ckpt_epoch, checkpoint['training_time_seconds']))
        net = UNet( ch=128, out_ch=3, ch_mult=(1,1,2,2,4,4), num_res_blocks=2, attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True, in_channels=3, resolution=256)
        net.load_state_dict(checkpoint['model_state_dict'])
        net = net.to(rank)
    except:
        raise Exception('No valid model found')
    
    # Generation
    time0 = time.time()
    X_gen = fixed_sampling(net, (n, 3, 256, 256), T, Alpha, Alpha_bar, Sigma,rank,START)
    #print('generated %s samples at epoch %s in %s seconds' % (n, ckpt_epoch, int(time.time() - time0)))

    # Save generated images
    #for i in range(n):
    save_image(rescale(X_gen[0]), os.path.join(output_directory, str(rank)+'_img_{}.jpg'.format(EPSILON)))
    #print('saved generated samples at epoch %s' % ckpt_epoch)
    return X_gen
    
def worker(rank, start, end,START):
    results = []
    for i in tqdm(range(start, end+1),disable = rank != 1):
        results.append(torch.reshape(generate('/s/b/proj/dkott_proj/newer_celebA_diffusion/output/', '/s/b/proj/dkott_proj/newer_celebA_diffusion/output/', 200, IMAGES_PER_GPU, 1_000, 0.0001, 0.02, {}, rank,START),[IMAGES_PER_GPU,-1]) )
    torch.save(torch.cat(results, dim=0), f'junk/tensor_{rank}.pt')

if __name__ == '__main__':
    START = -1
    parser = argparse.ArgumentParser(description='Set global variables via command line.')
    
    # Add an argument for the START variable
    parser.add_argument('--start', type=int, default=START, help='Starting point for the sampling process',required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Set the global variable START to the value provided from the command line
    START = args.start
    print(START)
    num_gpus = 16
    num_images = 10_000//IMAGES_PER_GPU
    images_per_process = num_images // num_gpus

    # Start the worker processes
    processes = []
    for rank in range(num_gpus):
        start = rank * images_per_process
        end = start + images_per_process if rank != num_gpus - 1 else num_images
        p = multiprocessing.Process(target=worker, args=(rank, start, end,START))
        p.start()
        processes.append(p)
    # Wait for all the worker processes to finish
    for p in processes:
        p.join()
    # Concatenate all the tensors
    tensors = [torch.load(f'junk/tensor_{i}.pt', map_location='cpu').cpu() for i in range(num_gpus) ]
    result = torch.cat(tensors, dim=0)
    
    torch.save(result,'all_images_'+str(START))

    print(result.shape)
