import torch
import timm
from PIL import Image
from torchvision import transforms

# load EfficientNet model
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=38)

checkpoint = torch.load("plant_disease_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open("data/imgs/00020.jpg")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output)

print("Predicted disease class index:", pred.item())