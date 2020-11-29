
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import pickle as pk
import matplotlib.pyplot as plt
from models import lenet
from models.brevitas_models import quant_lenet
from quantizer import util
import numpy as np


def train(model, criterion, optimizer, train_loader, val_loader, save_path, epoch=128, use_gpu=False):
    def validate(model, criterion, val_loader, use_gpu=False):
        val_size = len(val_loader.dataset)
        val_loss = 0
        correct = 0
        device = torch.device("cuda:0" if use_gpu else "cpu")

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # forward + backward + optimize
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels)

                val_loss += loss * inputs.size(0)

                # val accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            val_loss = val_loss/val_size
            val_accuracy = correct/val_size

        return val_loss, val_accuracy

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

        if(epoch_val_accuracy > 0.9):
            torch.save(model.state_dict(), save_path + f'e{epoch}.pth')

        # print([(n, m)
        #        for n, m in model.named_parameters() if 'quantizer' in n])

    return history


def test(model, criterion, test_loader, use_gpu=False):
    test_size = len(test_loader.dataset)
    device = torch.device("cuda:0" if use_gpu else "cpu")
    test_loss = 0.0
    test_accuracy = 0
    correct = 0

    model.eval()

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

    print(f'test_loss: {test_loss}, test_accuracy: {test_accuracy}')

    return test_loss, test_accuracy


def save_params(history, epoch, save_path, lr=0, momentum=0):
    with open(save_path + 'train_prams.pkl', 'wb') as f:
        pk.dump({
            'history': history,
            'num_of_epoch': epoch,
            'lr': lr,
            'momentum': momentum
        }, f)


def load_mnist():
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, ), std=(0.5,))
        ]
    )

    train_set = torchvision.datasets.MNIST(
        root='./data/MNIST', download=True, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        root='./data/MNIST', download=True, train=False, transform=transform)

    # split the training set and validation set
    torch.manual_seed(50)
    val_size = 5000
    train_size = len(train_set) - val_size
    test_size = len(test_set)

    batch_size = 32

    train_ds, val_ds = random_split(train_set, [train_size, val_size])

    trainLoader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False)

    testLoader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    print(
        f'train_set: {train_size}, valid_set: {val_size}, test_set: {test_size}')

    return trainLoader, val_loader, testLoader


def train_original():
    train_loader, val_loader, test_loader = load_mnist()

    # setup
    use_gpu = torch.cuda.is_available()
    epoch = 10
    lr = 0.001
    save_path = './results/lenet/lenet-'
    weights_save_path = './results/lenet/lenet-weights-'

    # model and optimizer
    model = lenet.LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = train(model, criterion, optimizer, train_loader,
                    val_loader, weights_save_path, epoch, use_gpu)

    test_loss, test_accuracy = test(
        model, criterion, test_loader, torch.cuda.is_available())
    history['test_loss'] = test_loss
    history['test_accuracy'] = test_accuracy

    save_params(history, epoch, save_path, lr)


def train_lsq():
    train_loader, val_loader, test_loader = load_mnist()

    # setup
    use_gpu = torch.cuda.is_available()
    epoch = 3
    lr = 0.001
    bits = 8

    save_path = f'./results/lenet/lenet-lsq-w{bits}a{bits}-'
    weights_save_path = f'./results/lenet/lenet-lsq-w{bits}a{bits}-weights-'

    # model and optimizer
    model = lenet.LeNet5()
    model.load_state_dict(torch.load('./weights/lenet-weights-e10.pth'))

    model = lenet.QuantLeNet5(model, bits)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = train(model, criterion, optimizer, train_loader,
                    val_loader, weights_save_path, epoch, use_gpu)

    test_loss, test_accuracy = test(
        model, criterion, test_loader, torch.cuda.is_available())
    history['test_loss'] = test_loss
    history['test_accuracy'] = test_accuracy

    save_params(history, epoch, save_path, lr)

    print([(n, m) for n, m in model.named_parameters()])


def train_brevitas():
    train_loader, val_loader, test_loader = load_mnist()

    # setup
    use_gpu = torch.cuda.is_available()
    epoch = 2
    lr = 0.001
    bits = 8

    save_path = f'./results/lenet/lenet-lsq-w{bits}a{bits}-'
    weights_save_path = f'./results/lenet/lenet-lsq-w{bits}a{bits}-weights-'

    weights = torch.load('./weights/lenet-weights-e10.pth')

    # # model and optimizer
    model = quant_lenet.QuantLeNet()
    # print(model)
    # model.load_state_dict(torch.load('./weights/lenet-weights-e10.pth'))

    # model = lenet.QuantLeNet5(model, bits)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = train(model, criterion, optimizer, train_loader,
                    val_loader, weights_save_path, epoch, use_gpu)

    test_loss, test_accuracy = test(
        model, criterion, test_loader, torch.cuda.is_available())
    history['test_loss'] = test_loss
    history['test_accuracy'] = test_accuracy

    # save_params(history, epoch, save_path, lr)

    # print([(n, m) for n, m in model.named_parameters()])


if __name__ == "__main__":
    # train_lsq()
    train_brevitas()
