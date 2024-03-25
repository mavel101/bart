#!/usr/bin/python
# -- coding: utf-8 --

#%%
import numpy as np
import matplotlib.pyplot as plt
from bart import bart

#%%
data = np.load("test_dft/data.npy")
traj = np.load("test_dft/traj.npy")
sens = np.load("test_dft/sens.npy")

data = data.T[np.newaxis,:,np.newaxis]
traj = traj.T

# %%
pics_config = 'pics -g --gpu-gridding -S -e -l1 -r 0.001 -i 50 -t'
nufft_config = 'nufft -g -i -d 140:140:1'

img = bart(1, pics_config , traj, data, sens)
# img = bart(1, nufft_config , traj, data)

img_abs = abs(img)

# %%
plt.figure()
plt.imshow(img_abs)
# %%
