import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from models.brevitas_models import quant_resnet


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


def train(model, criterion, optimizer, train_loader, val_loader, lr_scheduler, epoch=128, use_gpu=False):

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
        correct = 0

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

        lr_scheduler.step()
        t2 = time.time()
        t = t2 - t1

        # save the model
        PATH = f'./drive/My Drive/Colab Notebooks/checkpoints/resnet20-brevitas-w8a8-e{epoch + 1}.pth'
        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), }, PATH)

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

    model = quant_resnet.resnet20(no_quant=True, bit_width=8)
    criterion = nn.CrossEntropyLoss()
    lr = 0.10
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 200
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], last_epoch=-1)

    print(f'torch.cuda.is_available()): {torch.cuda.is_available()}')
    history = train(model, criterion, optimizer, train_loader, val_loader,
                    lr_scheduler, epoch=epochs, use_gpu=torch.cuda.is_available())
