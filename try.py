import numpy as np
import torch
import pdb

# now = np.load('/mnt/Data/myh/SLCA/pretrained/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
now = torch.load('now.pth')
old = torch.load('/home/myh/timm_vit-B-16_pretrained.pth')

dd = {}
sum = {}
for key in now.keys():
    dd[key] = (now[key].detach()-old[key]).abs()
    sum[key] = dd[key].reshape(-1).sum()/dd[key].numel()
pdb.set_trace()