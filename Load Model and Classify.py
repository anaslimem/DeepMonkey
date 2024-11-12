import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import gradio as gr

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('best_model.pth', map_location=device)
model = model.to(device)

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
    image = image.to(device)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    return classes[predicted.item()]
def img(image):
    return classify(model,image_transforms,image,classes)
examples=[
    ["Test this picture/Patas-Monkey.jpg"],
    ["Test this picture/pygmy.jpeg"],
    ["Test this picture/images.jpeg"]
]
demo=gr.Interface(
    fn=img,
    inputs=[gr.Image(type="filepath",label="Your Image")],
    outputs=[gr.Text(label="Monkey Species")],
    title="Monkey Species classification ",
    examples=examples)
demo.launch(share=True)

