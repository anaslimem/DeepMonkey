import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image

classes = [
    'Mantled_Howler',
    'Patas_Monkey',
    'Bald_Uakari',
    'Japanese_Macaque',
    'Pygmy_Marmoset',
    'White_Headed_Capuchin',
    'Silvery_Marmoset',
    'Common_Squirrel_Monkey',
    'Black_Headed_Night_Monkey',
    'Nilgiri_Langur',
]
model = torch.load('best_model.pth')

mean = [0.4363, 0.4328, 0.3291]
std = [0.2129, 0.2075, 0.2037]
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])


def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    image = image.to('cuda')  # move image to the GPU

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    print(classes[predicted.item()])

classify(model, image_transforms,"./Test this picture/pygmy.jpeg", classes)


