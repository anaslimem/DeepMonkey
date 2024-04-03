import os
import torch
import torchvision
import torchvision.transforms as transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# path of the training dataset
os.listdir('./Monkeys/training/training')
train_ds_path = './Monkeys/training/training'
# Training Transformer
train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# Training dataset
train_ds = torchvision.datasets.ImageFolder(root=train_ds_path, transform=train_transform)
# Manage loading data
train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=32, shuffle=False)


# Calculate the mean and standard deviation of the image dataset

def get_mean_std(loader):
    mean = 0
    std = 0
    total_img_count = 0
    for images, _ in loader:
        # Move images to the GPU
        images = images.to(device)
        img_count_in_a_batch = images.size(0)
        images = images.view(img_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_img_count += img_count_in_a_batch
    mean /= total_img_count
    std /= total_img_count
    return mean, std


mean, std = get_mean_std(train_loader)
print("Mean:", mean)
print("Standard Deviation:", std)
