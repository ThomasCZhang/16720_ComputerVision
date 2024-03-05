import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import *

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]


width, height = GetRectShape(rect)

all_rects = np.zeros((seq.shape[2], 4))
all_rects[0] = np.array(rect)

It = seq[:, :, 0]
for i in range(seq.shape[2]-1):
    It1 = seq[:, :, i+1]

    if i % 20 == 0:
        fig, ax = plt.subplots(1,1)
        ax.imshow(seq[:, :, i], cmap = 'gray')
        ax.add_patch(patches.Rectangle((rect[0], rect[1]), width, height, facecolor="none", ec='r', lw=2))
        ax.set_xticks([])
        ax.set_yticks([])        
        # plt.show()
        fig.savefig(f"../results/girl_wcrt_{i}.png")

    p = LucasKanade(It, It1, rect, threshold, num_iters)

    p0_s = np.array([rect[0]-all_rects[0,0]+p[0], rect[1]-all_rects[0, 1]+p[1]])
    p_s = LucasKanade(seq[:, :, 0], It1, all_rects[0, :], threshold, num_iters, p0_s)
    
    # Check if the current template is good
    if np.linalg.norm((p_s-p0_s)-p) < template_threshold: # If the template is good update based on p*
        # print(f'{i}: Normal')
        p_s = p_s-p0_s
        rect = [rect[0] + p_s[0], rect[1] + p_s[1], rect[2] + p_s[0], rect[3] + p_s[1]]
        It = It1.copy()
        p0 = np.zeros(2)
    else: # If the current template is bad. We don't update rect.
        # print(f'{i}: Keep Old Template')
        all_rects[i, :] = all_rects[i-1, :].copy()
        It = seq[:, :, i-1]
        p0 = p_s - p0_s

    # if np.linalg.norm(p_s-p) < template_threshold:
    #     rect = [rect[0] + p[0], rect[1] + p[1],  rect[2] + p[0], rect[3] + p[1]]
    #     It = It1.copy()
    # else:
    #     rect = all_rects[0, :]
    #     rect = [rect[0] + p_s[0], rect[1] + p_s[1], rect[2] + p_s[0], rect[3] + p_s[1]]
    #     It = It1.copy()

    all_rects[i+1] = np.array(rect)
    
np.save("../results/girlseqrects-wcrt.npy", all_rects)