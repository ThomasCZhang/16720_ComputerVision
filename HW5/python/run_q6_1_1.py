from torch.utils.data import Dataset, DataLoader, TensorDataset
# class MNIST(Dataset):
#     def __init__(self,data, labels) -> None:
#         self.data = torch.from_numpy(data).float()
#         self.labels = torch.from_numpy(labels).float()
    
#     def __getitem__(self, index) -> tuple:
#         return self.data[index], self.labels[index]
    
#     def __len__(self) -> int:
#         return self.data.shape[0]
    
if __name__ == '__main__':
    from nn import get_random_batches
    import torch
    import scipy
    import matplotlib.pyplot as plt
    import torch.nn as nn            # containing various building blocks for your neural networks

    from time import time

    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

    train_x, train_y = train_data['train_data'], train_data['train_labels']
    valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

    n_example, n_dim = train_x.shape
    n_example, n_class = train_y.shape

    # trainset = MNIST(train_x, train_y)
    # trainset_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()), batch_size=128, shuffle=True, num_workers=2)

    # testset = MNIST(valid_x, valid_y)
    # testset_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    max_iters = 100
    batch_size = 128
    learning_rate = 2e-2
    hidden_size = 64

    batches = get_random_batches(train_x,train_y,batch_size)
    batch_num = len(batches)
    
    model = torch.nn.Sequential(
        nn.Linear(n_dim, hidden_size),
        nn.Sigmoid(),
        nn.Linear(hidden_size, n_class),
        # nn.Softmax(1)
    )

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    device = 'cpu'
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)

    loss_rec = []
    acc_rec = []
    valid_loss_rec = []
    valid_acc_rec = []
    valid_x, valid_y = torch.from_numpy(valid_x).float(), torch.from_numpy(valid_y).float()
    for ep in range(max_iters):
        avg_loss = 0
        accuracy = 0
        model.train()
        for (data, target) in batches:
            data, target = torch.from_numpy(data).float(), torch.from_numpy(target).float()
            # forward pass
            output = model(data)

            loss = loss_fn(output, target)
            # backward pass
            optimizer.zero_grad() # clear the gradients of all tensors being optimized.
            loss.backward() # accumulate (i.e. add) the gradients from this forward pass
            optimizer.step() # performs a single optimization step (parameter update)
            
            loss_ = loss.item()
            avg_loss += loss_

            _, preds = torch.max(output, 1)
            _, labels = torch.max(target, 1)
            accuracy += torch.sum(preds == labels)
        accuracy = accuracy/n_example
        avg_loss = avg_loss/batch_num
        loss_rec.append(avg_loss)
        acc_rec.append(accuracy)

        model.eval()
        output = model(valid_x)
        loss = loss_fn(output, valid_y)
        valid_loss = loss.item()
        valid_loss_rec.append(valid_loss)
        _, preds = torch.max(output, 1)
        _, labels = torch.max(valid_y, 1)  
        valid_acc = torch.sum(preds == labels)/valid_x.shape[0] 
        valid_acc_rec.append(valid_acc)     
        print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}, Validation Accuracy: {:.3f}".format(ep+1, max_iters, avg_loss, accuracy, valid_acc))

    fig, ax = plt.subplots(1)
    ax.plot(loss_rec, label = 'Training')
    ax.plot(valid_loss_rec, label = 'Validation')
    ax.set_title('Loss vs Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig('../results/6_1_loss.png')
    plt.close(fig)

    fig, ax = plt.subplots(1)
    ax.plot(acc_rec, label = 'Training')
    ax.plot(valid_acc_rec, label = 'Validation')
    ax.set_title('Accuracy vs Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.savefig('../results/6_1_acc.png')
    plt.close(fig)
