import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 128
learning_rate = 2e-3
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
n_example, n_dim = train_x.shape
n_example, n_class = train_y.shape
initialize_weights(n_dim, hidden_size, params, 'layer1')
initialize_weights(hidden_size, n_class, params, 'output')
##########################

init_layer1 = params['Wlayer1'].copy()

# with default settings, you should get loss < 150 and accuracy > 80%
train_acc_list = []
train_loss_list = []
valid_acc_list = []
valid_loss_list = []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # pass
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        h1 = forward(xb, params, 'layer1', sigmoid)
        probs = forward(h1, params, 'output', softmax)

        loss, acc = compute_loss_and_acc(yb, probs)
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss/batch_size
        total_acc += acc*xb.shape[0]/n_example

        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        delta3 = backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        for name in ['layer1', 'output']:
            params['W' + name] -= learning_rate*params['grad_W' + name]
            params['b' + name] -= learning_rate*params['grad_b' + name]
        ##########################
    # append the training loss and accuracy
    train_loss_list.append(total_loss)
    train_acc_list.append(total_acc)

    h1 = forward(valid_x, params, 'layer1', sigmoid)
    probs = forward(h1, params, 'output', softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    valid_loss_list.append(loss/batch_size)
    valid_acc_list.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
ax.plot(train_loss_list, label = 'Training Loss')
ax.plot(valid_loss_list, label = 'Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'Loss vs Epoch. Learning Rate = {learning_rate}')
ax.legend()
# fig.savefig(f'../results/3_2_loss{learning_rate}.png')
plt.close(fig)

fig, ax = plt.subplots(1)
ax.plot(train_acc_list, label = 'Training Accuracy')
ax.plot(valid_acc_list, label = 'Validation Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title(f'Accuracy vs Epoch. Learning Rate = {learning_rate}')
ax.legend()
# fig.savefig(f'../results/3_2_accuracy{learning_rate}.png')
plt.close(fig)

# run on validation set and report accuracy! should be above 75%
# valid_acc = None
##########################
##### your code here #####
import pickle
with open('q3_weights.pickle', 'rb') as file:
    params = pickle.load(file)

h1 = forward(valid_x, params, 'layer1', sigmoid)
probs = forward(h1, params, 'output', softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
#########################

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
# saved_params = {k:v for k,v in params.items() if '_' not in k}
# with open('q3_weights.pickle', 'wb') as handle:
#     pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
# import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

with open('q3_weights.pickle', 'rb') as file:
    params = pickle.load(file)
# visualize weights here
##########################
##### your code here #####
fig = plt.figure()
fig.suptitle('First Layer Weights Before Training')
grid=ImageGrid(fig, 111, (8,8))

for i, row in enumerate(init_layer1.T):
    grid[i].imshow(row.reshape(32,32))
    grid[i].set_xticks([0,16])
    grid[i].set_yticks([0,16])
# fig.savefig('../results/3_3_initial.png')
plt.show()
plt.close(fig)

fig = plt.figure()
fig.suptitle('First Layer Weights After Training')
grid=ImageGrid(fig, 111, (8,8))

for i, row in enumerate(params['Wlayer1'].T):
    grid[i].imshow(row.reshape(32,32))
    grid[i].set_xticks([0,16])
    grid[i].set_yticks([0,16])

# fig.savefig('../results/3_3_trained.png')
plt.show()
plt.close(fig)
##########################
# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
h1 = forward(valid_x, params, 'layer1', sigmoid)
probs = forward(h1, params, 'output', softmax)
labels = np.argmax(valid_y, axis = 1)
preds = np.argmax(probs, axis = 1)
for i,j in zip(labels, preds):
    confusion_matrix[i,j] += 1
##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.title('Confusion Matrix of Validation Data')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.savefig('../results/3_4.png')
plt.show()