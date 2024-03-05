import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
import matplotlib.pyplot as plt
from torchsummary import summary
import os
from PIL import Image

class CNNModel(nn.Module):
    def __init__(self):
        #TODO: Define Features
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace= True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace= True),
            nn.Dropout(0.2),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace= True),
            nn.Dropout(0.2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace= True)
        )
        
        #TODO: Define Classifiers
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(15*15*256, 17),
            # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            # nn.ReLU(inplace = True),
            # nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            # nn.ReLU(inplace = True),
            # nn.Conv2d(256, 8, kernel_size=(1, 1), stride=(1, 1)),
        )
        
        for neuron in self.features:
            if isinstance(neuron,nn.Conv2d):
                nn.init.xavier_uniform_(neuron.weight)
        
        
    def forward(self, x):
        #TODO: Define forward pass
        x = self.features(x)
        # print(x.shape)
        x = self.classifier(x)
        # x = F.linear
        # x = F.max_pool2d(x, (x.shape[2], x.shape[3]))
        return x 
    pass

def GetData(your_path: str):
    train_transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # train_file_paths = open(os.path.join(your_path, 'train_files.txt')).read().splitlines()
    # train_labels = open(os.path.join(your_path, 'train_labels.txt')).read().splitlines()
    # test_file_paths = open(os.path.join(your_path, 'test_files.txt')).read().splitlines()
    # test_labels = open(os.path.join(your_path, 'test_labels.txt')).read().splitlines()
    
    trainset = torchvision.datasets.ImageFolder(root= os.path.join(your_path, 'train'),
                                           transform=train_transform)
    testset = torchvision.datasets.ImageFolder(root= os.path.join(your_path, 'test'),
                                           transform=train_transform)

    # ['aquarium', 'desert', 'highway', 'kitchen', 'laundromat', 'park', 'waterfall', 'windmill']
    return trainset, testset

def train(dataloader, device, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    current = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes = 17)
        y_one_hot = y_one_hot.float()
        
        # Compute prediction error
        pred = model(X)

        # final_layer=nn.MaxPool2d((pred.size(2),pred.size(3)))
        # pred=final_layer(pred)
        pred=torch.reshape(pred,(-1,17))#(-1,8)
        pred=F.sigmoid(pred)

        loss = loss_fn(pred, y_one_hot)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()       

        if batch % 2 == 0:
            loss_ = loss.item()
            current += len(X)
            print(f"\rloss: {loss_:>7f}  [{current:>5d}/{size:>5d}]", end = '')
            
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()
    

    train_loss /= num_batches
    correct /= size
    print(f"\rloss: {train_loss:>7f}  Accuracy: {100*correct:>7f}%", end = '')

    return correct, train_loss

def test(dataloader, device, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_one_hot = torch.nn.functional.one_hot(y, num_classes = 17)
            y_one_hot = y_one_hot.float()
            pred = model(X)

            # final_layer=nn.MaxPool2d((pred.size(2),pred.size(3)))
            # pred=final_layer(pred)
            pred=torch.reshape(pred,(-1,17))#(-1,8)
            pred=F.sigmoid(pred)

            test_loss += loss_fn(pred, y_one_hot).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"\nTest: Avg loss: {test_loss:>7f}  Accuracy: {(100*correct):>7f}% ")
    return correct, test_loss

def main():
    device = 'cuda'
    num_epoch = 20

    # model = CNNModel()
    # model.to(device)
    # loss_fn=nn.BCELoss()
    # optimizer=torch.torch.optim.SGD(model.parameters(), lr=0.1, momentum = 0.9)

    trainset, testset = GetData('..\\data\\oxford-flowers17')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    # acc_rec = []
    # loss_rec = []
    # valid_acc_rec = []
    # valid_loss_rec = []
    # for i in range(num_epoch):
    #     acc, loss = train(trainloader, device, model, loss_fn, optimizer)
    #     valid_acc, valid_loss = test(testloader, device, model, loss_fn)

    #     acc_rec.append(acc)
    #     loss_rec.append(loss)
    #     valid_acc_rec.append(valid_acc)
    #     valid_loss_rec.append(valid_loss)
        
    #     torch.save({'acc': acc_rec, 'loss': loss_rec, 'val_acc': valid_acc_rec, 'val_loss': valid_loss_rec}, '../6_2/cnn_record.pt')
    #     torch.save(model.state_dict(), '../6_2/cnn_model.pt')
    #     torch.save(optimizer.state_dict(), '../6_2/cnn_optimizer.pt')
    #     if acc > 0.97: # Overfitting
    #         break

    # fig, ax = plt.subplots(1)
    # ax.plot(loss_rec, label = 'Training')
    # ax.plot(valid_loss_rec, label = 'Validation')
    # ax.set_title('Loss vs Epochs')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Loss')
    # ax.legend()
    # fig.savefig('../results/6_2_cnnloss.png')
    # plt.close(fig)

    # fig, ax = plt.subplots(1)
    # ax.plot(acc_rec, label = 'Training')
    # ax.plot(valid_acc_rec, label = 'Validation')
    # ax.set_title('Accuracy vs Epochs')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Accuracy')
    # ax.legend()
    # fig.savefig('../results/6_2_cnnacc.png')
    # plt.close(fig)

    squeezenet = torchvision.models.squeezenet1_1(weights = 'DEFAULT')
    squeezenet.classifier[1] = nn.Conv2d(512, len(trainset.classes), kernel_size=1)
    squeezenet.num_classes = trainset.classes
    squeezenet.to(device)


    acc_rec = []
    loss_rec = []
    valid_acc_rec = []
    valid_loss_rec = []
    loss_fn = nn.CrossEntropyLoss()
    optimizer=torch.torch.optim.SGD(squeezenet.parameters(), lr=0.5, momentum = 0.9)

    for i in range(num_epoch):
        acc, loss = train(trainloader, device, squeezenet, loss_fn, optimizer)
        valid_acc, valid_loss = test(testloader, device, squeezenet, loss_fn)

        acc_rec.append(acc)
        loss_rec.append(loss)
        valid_acc_rec.append(valid_acc)
        valid_loss_rec.append(valid_loss)
        
        torch.save({'acc': acc_rec, 'loss': loss_rec, 'val_acc': valid_acc_rec, 'val_loss': valid_loss_rec}, '../6_2/squeeze_record.pt')
        torch.save(squeezenet.state_dict(), '../6_2/squeeze_model.pt')
        torch.save(optimizer.state_dict(), '../6_2/squeeze_optimizer.pt')
        if acc > 0.97: # Overfitting
            break

    fig, ax = plt.subplots(1)
    ax.plot(loss_rec, label = 'Training')
    ax.plot(valid_loss_rec, label = 'Validation')
    ax.set_title('Loss vs Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig('../results/6_2_squeezeloss.png')
    plt.close(fig)

    fig, ax = plt.subplots(1)
    ax.plot(acc_rec, label = 'Training')
    ax.plot(valid_acc_rec, label = 'Validation')
    ax.set_title('Accuracy vs Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    fig.savefig('../results/6_2_squeezeacc.png')
    plt.close(fig)

if __name__ == '__main__':
    main()