import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 0
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(32, 32, params, 'layer2')
initialize_weights(32, 32, params, 'layer3')
initialize_weights(32, 1024, params, 'output')
##########################

# should look like your previous training loops
all_loss = []
for itr in range(max_iters):
    total_loss = 0
    # print(itr)
    # i = 0
    for xb,_ in batches:

        # xb = (xb - np.min(xb))/(np.max(xb)-np.min(xb))
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        # pass
        # forward
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'layer2', relu)
        h3 = forward(h2, params, 'layer3', relu)
        y = forward(h3, params, 'output', sigmoid)

        loss = np.sum(np.power(xb - y, 2))/batch_size

        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss

        # backward
        delta1 = 2*(y-xb)
        delta2 = backwards(delta1, params, 'output', sigmoid_deriv)
        delta3 = backwards(delta2, params, 'layer3', relu_deriv)
        delta4 = backwards(delta3, params, 'layer2', relu_deriv)
        delta5 = backwards(delta4, params, 'layer1', relu_deriv)

        # apply gradient
        ##########################
        ##### your code here #####
        for name in ['layer1', 'layer2', 'layer3', 'output']:
            for w_name in ['W', 'b']:
                param_name = w_name+name
                params['m_' + param_name] = 0.9*params['m_' + param_name] - learning_rate*params['grad_'+param_name]
                params[param_name] += params['m_' + param_name] 
        ##########################
    all_loss.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1)
# ax.plot(all_loss)
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss')
# ax.set_title('Loss vs Epoch for Momentum Model')
# fig.savefig('../results/5_2.png')
# plt.show()

# import pickle
# saved_params = {k:v for k,v in params.items() if '_' not in k}
# with open('q5_weights.pickle', 'wb') as handle:
#     pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

import pickle
with open('q5_weights.pickle', 'rb') as file:
    params = pickle.load(file)

# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
import string
letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
valid_labels = valid_data['valid_labels']
valid_labels = np.argmax(valid_labels, axis = 1)
classes = [0, 5, 10, 15, 20]
for i in classes:
    idxs = np.where(valid_labels == i)[0][:2]# First two images from a class
    ims = valid_x[idxs, :]

    h1 = forward(ims, params, 'layer1', relu)
    h2 = forward(h1, params, 'layer2', relu)
    h3 = forward(h2, params, 'layer3', relu)
    y = forward(h3, params, 'output', sigmoid)

    fig, ax = plt.subplots(2,2)
    for j in range(2):
        ax[0, j].imshow(ims[j].reshape(32,32).T, cmap = 'gray')
        ax[0, j].set_title(f'Original Image: Class {letters[i]}')
        ax[0, j].set_xticklabels([])
        ax[1, j].imshow(y[j].reshape(32,32).T, cmap = 'gray')
        ax[1, j].set_title(f'Reconstructed Image: Class {letters[i]}')
    fig.savefig(f'../results/5_3_1{letters[i]}.png')

##########################


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR 
##########################
##### your code here #####
h1 = forward(valid_x, params, 'layer1', relu)
h2 = forward(h1, params, 'layer2', relu)
h3 = forward(h2, params, 'layer3', relu)
y = forward(h3, params, 'output', sigmoid)

all_psnr = [peak_signal_noise_ratio(valid_x[i, :].reshape(32,32).T, y[i,:].reshape(32,32).T) for i in range(valid_x.shape[0])]
print(f'Mean PSNR: {np.mean(all_psnr)}')
##########################
