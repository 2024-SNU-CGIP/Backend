import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from model import FusionModelMobileNetV2
from dataset import DentalDataset

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a transformation for the photos (RGB images)
photo_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a transformation for the X-rays (grayscale images)
xray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Define paths and labels
photo_paths_L = []
for i in range(1, 111):
    photo_paths_L.append(f"images/TrainData/Extraction/Photo_L/{i}_L.jpg")
for i in range(1, 111):
    photo_paths_L.append(f"images/TrainData/Non_Extraction/Photo_L/{i}_L.jpg")
photo_paths_U = []
for i in range(1, 111):
    photo_paths_U.append(f"images/TrainData/Extraction/Photo_U/{i}_U.jpg")
for i in range(1, 111):
    photo_paths_U.append(f"images/TrainData/Non_Extraction/Photo_U/{i}_U.jpg")

xray_paths = []
for i in range(1, 111):
    xray_paths.append(f"images/TrainData/Extraction/Xray/{i}_lat.jpg")
for i in range(1, 111):
    xray_paths.append(f"images/TrainData/Non_Extraction/Xray/{i}_lat.jpg")

labels = [1] * 110 + [0] * 110

# Create an instance of the dataset
# Create an instance of the dataset with separate transforms
dataset = DentalDataset(photo_paths_L, photo_paths_U, xray_paths, labels,
                        photo_transform=photo_transform,
                        xray_transform=xray_transform)


# Split the dataset into training, validation, and test sets
train_val_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.1, shuffle=True, stratify=labels)
train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, shuffle=True, stratify=[labels[i] for i in train_val_indices])

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)
test_subset = Subset(dataset, test_indices)

# DataLoader for training, validation, and test
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)

# Define the loss function and other hyperparameters
criterion = lambda x, y: nn.BCELoss()(x, y) + 0.5 * nn.MSELoss()(x, y)

# Adjust learning rate and add weight decay for L2 regularization
learning_rate = 0.0005
weight_decay = 1e-3  # 기존 1e-4에서 증가
num_epochs = 1000  # Reduce the number of epochs
patience =1000  # Decrease patience for early stopping

# Instantiate the model and optimizer
model = FusionModelMobileNetV2(num_classes=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Early stopping variables
best_val_loss = float('inf')
epochs_without_improvement = 0

# Add learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience):
    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for photos_L, photos_U, xrays, labels in train_loader:
            photos_L, photos_U, xrays, labels = photos_L.to(device), photos_U.to(device), xrays.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(photos_L, photos_U, xrays).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'model_weights.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    model.load_state_dict(torch.load('model_weights.pth'))

def evaluate_model(model, data_loader, device):
    model.eval()
    test_accuracy = 0

    with torch.no_grad():
        for photos_L, photos_U, xrays, labels in data_loader:
            photos_L, photos_U, xrays, labels = photos_L.to(device), photos_U.to(device), xrays.to(device), labels.to(device)
            outputs = model(photos_L, photos_U, xrays).squeeze()
            test_accuracy += ((outputs > 0.5) == labels).sum().item()
    
    return test_accuracy / len(data_loader.dataset)