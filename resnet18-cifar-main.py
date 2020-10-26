
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import models.resnet as resnet
import pickle
import matplotlib.pyplot as plt


def get_train_accuracy(model, criterion, train_loader, use_gpu=False):
    train_size = len(train_loader.dataset)
    correct = 0
    device = torch.device("cuda:0" if use_gpu else "cpu")

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            outputs = model(inputs).to(device)

            # val accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        print(correct)
        return correct/train_size


def test(model, criterion, test_loader, use_gpu=False):
    test_size = len(test_loader.dataset)
    device = torch.device("cuda:0" if use_gpu else "cpu")
    test_loss = 0.0
    test_accuracy = 0
    correct = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)

            test_loss += loss * inputs.size(0)

            # val accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        test_loss = test_loss/test_size
        test_accuracy = correct/test_size

    return test_loss, test_accuracy


def train(model, criterion, optimizer, train_loader, val_loader, epoch=128, use_gpu=False):

    train_size = len(train_loader.dataset)
    device = torch.device("cuda:0" if use_gpu else "cpu")

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    model = model.to(device)

    for epoch in range(epoch):  # loop over the dataset multiple times
        running_loss = 0.0

        print(f'------------------------------\n Epoch: {epoch + 1}')

        t1 = time.time()
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # as loss.item() return the average batch loss, so convert it to the total loss
            running_loss += loss.item() * inputs.size(0)

        t2 = time.time()
        t = t2 - t1

        # save the model
        PATH = f'./model_weights/resnet18-e{epoch + 1}.pth'
        torch.save(model.state_dict(), PATH)

        epoch_train_loss = running_loss/train_size
        epoch_train_accuracy = get_train_accuracy(
            model, criterion, train_loader, use_gpu)
        epoch_val_loss, epoch_val_accuracy = validate(
            model, criterion, val_loader, use_gpu)
        print(f'time: {int(t)}sec train_loss: {epoch_train_loss}, train_accuracy: {epoch_train_accuracy}, val_loss: {epoch_val_loss}, val_accuracy: {epoch_val_accuracy}')

        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(epoch_train_accuracy)
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

    return history


def plot_result(history, epoch):

    num_of_epoch = range(1, epoch)

    plt.plot(num_of_epoch, history['train_loss'], label="train_loss")
    plt.plot(num_of_epoch, history['val_loss'], label="val_loss")
    plt.xlabel('epoch')
    plt.ylabel('crossEntropyLoss')
    plt.title('model loss')

    plt.savefig('./model_loss.jpg')

    plt.plot(num_of_epoch, history['train_accuracy'], label="train_accuracy")
    plt.plot(num_of_epoch, history['val_accuracy'], label="val_accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('model accuracy')
    plt.savefig('./model_acc.jpg')


def save_params(history, epoch, lr, momentum):
    with open('train_prams.pkl', 'wb') as f:
        pk.dump({
            'history': history,
            'num_of_epoch': epoch,
            'lr': lr,
            'momentum': momentum
        }, f)


if __name__ == "__main__":
    # load the data
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])

    ds = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=train_transform)

    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    # split the training set and validation set
    torch.manual_seed(50)
    test_size = len(test_ds)
    val_size = 5000
    train_size = len(ds) - val_size
    batch_size = 256

    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    classes = ds.classes

    # define a model
    epoch = 128
    lr = 0.003
    momentum = 0.9
    model = resnet.resnet_18_cifar()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    history = train(model, criterion, optimizer, train_loader,
                    val_loader, epoch=epoch, torch.cuda.is_available())

    test_loss, test_accuracy = test(
        model, criterion, test_loader, torch.cuda.is_available())
    history['test_loss'] = test_loss
    history['test_accuracy'] = test_accuracy

    save_params(history, epoch, lr, momentum)
    plot_result(history, epoch)
