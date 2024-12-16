import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from visualizeXAI import FusionModelMobileNetV2_XAI

class FusionModelMobileNetV2(nn.Module):
    def __init__(self, num_classes=1):
        super(FusionModelMobileNetV2, self).__init__()

        # MobileNetV2 for photo_L
        self.mobilenet_L = models.mobilenet_v2(pretrained=True)
        self.mobilenet_L.classifier = nn.Identity()  # Remove the final classifier

        # MobileNetV2 for photo_U
        self.mobilenet_U = models.mobilenet_v2(pretrained=True)
        self.mobilenet_U.classifier = nn.Identity()  # Remove the final classifier

        # MobileNetV2 for X-ray (modify first conv layer for grayscale input)
        self.mobilenet_xray = models.mobilenet_v2(pretrained=True)
        self.mobilenet_xray.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mobilenet_xray.classifier = nn.Identity()  # Remove the final classifier

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(1280 * 3, 1024),  # MobileNetV2 outputs 1280 features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, photo_L, photo_U, xray):
        # Extract features from each input
        features_L = self.mobilenet_L(photo_L)
        features_U = self.mobilenet_U(photo_U)
        features_xray = self.mobilenet_xray(xray)

        # Concatenate features
        combined_features = torch.cat((features_L, features_U, features_xray), dim=1)

        # Pass through fusion layers
        output = self.fusion(combined_features)

        return output

model = FusionModelMobileNetV2(num_classes=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModelMobileNetV2_XAI()
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model = model.to(device)
model.eval()

photo_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

xray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
