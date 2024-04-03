import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

train_ds_path = './Monkeys/training/training'
val_ds_path = './Monkeys/validation/validation'
mean = [0.4363, 0.4328, 0.3291]
std = [0.2129, 0.2075, 0.2037]
# Dataset preparation
train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize(torch.tensor(mean), torch.tensor(std))])
val_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(torch.tensor(mean), torch.tensor(std))])
train_ds = torchvision.datasets.ImageFolder(root=train_ds_path, transform=train_transform)
val_ds = torchvision.datasets.ImageFolder(root=val_ds_path, transform=val_transform)


def show_transformed_img(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print('labels: ', labels)
    plt.show()


show_transformed_img(train_ds)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)


# Moving to GPU using cuda
def set_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return torch.device(device)


def save_checkpoint(model, epoch, optimizer, best_acc):
    state= {
        'epoch': epoch +1 ,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best accuracy': best_acc
    }
    torch.save(state, f'model_best_checkpoints.pth.tar')


# Evaluation of the model
def evaluate_model_on_val_set(model, val_loader):
    model.eval()
    predicted_correctly_on_epochs = 0
    total = 0
    device = set_device()
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_correctly_on_epochs += (labels==predicted).sum().item()
    epoch_acc = 100 * predicted_correctly_on_epochs / total
    print("     -Testing dataset. Got %d out of %d images correctly (%.2f%%)" % (predicted_correctly_on_epochs, total, epoch_acc))
    return epoch_acc

#Traning the neural network
def train_nn(model, train_loader, val_loader, criterion, optimizer, n_epochs):
    device = set_device()
    best_acc = 0
    for epoch in range(n_epochs):
        print("Epoch number  %d " % (epoch+1))
        model.train()
        running_loss = 0
        running_corrects = 0
        total = 0
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += (labels == predicted).sum().item()
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100*running_corrects/total
        print("     -Training dataset. Got %d out of %d images correctly (%.2f%%).Epoch loss: %.3f" % (running_corrects, total, epoch_acc, epoch_loss))
        val_dataset_acc = evaluate_model_on_val_set(model, val_loader)
        if (val_dataset_acc > best_acc):
            best_acc = val_dataset_acc
            save_checkpoint(model, epoch, optimizer, best_acc)
    print("Finished")
    return model


resnet18_model = models.resnet18()
num_ftrs = resnet18_model.fc.in_features
num_of_classes = 10
resnet18_model.fc = nn.Linear(num_ftrs, num_of_classes)
device = set_device()
resnet18_model=resnet18_model.to(device)
checkpoint = torch.load('model_best_checkpoints.pth.tar')
resnet18_model.load_state_dict(checkpoint['model'])
torch.save(resnet18_model,"best_model.pth")
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)
train_nn(resnet18_model, train_loader, val_loader, loss_fn, optimizer, n_epochs=5)





