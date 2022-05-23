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
  G = legacy.load_network_pkl(f)['G_ema'].to(device)

# Labels.
label = torch.zeros([1, G.c_dim], device=device)

# Generate images.
for seed_idx, seed in enumerate(seeds):
  print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
  z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
  img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
  img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
  PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
