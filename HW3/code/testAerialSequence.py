import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanadeAffine import *
from SubtractDominantMotion import *
import time

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')

start = time.time()
for i in range(seq.shape[2]-1):
    print(i)
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]
    mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance)
    if (i+1)%30 == 0:
        y, x = np.nonzero(mask)
        fig, ax = plt.subplots(1,1)
        ax.imshow(It1, cmap = 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(x, y, 'bs', markersize = 1)
        fig.savefig(f'../results/inv_airplane_{i+1}.png')
print('Duration: ', time.time()-start)