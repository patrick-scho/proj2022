import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy

network_pkl = "networks/ffhq.pkl"
outdir = "out"
seeds = [600, 601]
truncation_psi = 1
noise_mode = "const"

print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
  D = legacy.load_network_pkl(f)['D'].to(device)

