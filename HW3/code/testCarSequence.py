import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import *
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")

rect = [59, 116, 145, 151]
width, height = GetRectShape(rect)

all_rects = np.zeros((seq.shape[2], 4))
all_rects[0] = np.array(rect)

p0 = np.zeros(2)
for i in range(seq.shape[2]-1):
    print(f'{i}', end = '\r')
    It = seq[:, :, i]
    It1 = seq[:, :, i+1]

    if i % 100 == 0:
        fig, ax = plt.subplots(1,1)
        ax.imshow(It, cmap = 'gray')
        ax.add_patch(patches.Rectangle((all_rects[i,0], all_rects[i,1]), width, height, facecolor="none", ec='r', lw=2))
        ax.set_xticks([])
        ax.set_yticks([])
        # plt.show()
        fig.savefig(f"../results/Car_{i}.png")

    p = LucasKanade(It, It1, rect, threshold, num_iters)
    rect = np.array([rect[0] + p[0], rect[1] + p[1],  rect[2] + p[0], rect[3] + p[1]])
    all_rects[i+1] = np.array(rect)
    
np.save("../results/carseqrects.npy", all_rects)


