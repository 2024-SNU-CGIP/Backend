from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import FusionModelMobileNetV2
from dataset import DentalDataset
from train import train
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

app = FastAPI()

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModelMobileNetV2(num_classes=1)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model = model.to(device)
model.eval()

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

num_extraction_images = 110
num_non_extraction_images = 110


@app.post("/predict")
async def predict(
    photo_L: UploadFile = File(...),
    photo_U: UploadFile = File(...),
    xray: UploadFile = File(...),
):
    try:
        # Read and preprocess the images
        photo_L_image = Image.open(BytesIO(await photo_L.read()))
        photo_U_image = Image.open(BytesIO(await photo_U.read()))
        xray_image = Image.open(BytesIO(await xray.read()))

        photo_L_tensor = photo_transform(photo_L_image).unsqueeze(0).to(device)
        photo_U_tensor = photo_transform(photo_U_image).unsqueeze(0).to(device)
        xray_tensor = xray_transform(xray_image).unsqueeze(0).to(device)

        # Make a prediction
        with torch.no_grad():
            prediction = model(photo_L_tensor, photo_U_tensor, xray_tensor)

        # Return the result
        return JSONResponse(content={"prediction": prediction.item()})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/upload_data")
async def upload_data(
    label: int,
    photo_L: UploadFile = File(...),
    photo_U: UploadFile = File(...),
    xray: UploadFile = File(...)
):
    try:
        # Save the images
        photo_L_image = Image.open(BytesIO(await photo_L.read()))
        photo_U_image = Image.open(BytesIO(await photo_U.read()))
        xray_image = Image.open(BytesIO(await xray.read()))

        photo_L_image.save(f"images/TrainData/Extraction/Photo_L/{num_extraction_images + 1}_L.jpg" if label == 1 else f"images/TrainData/Non_Extraction/Photo_L/{num_non_extraction_images + 1}_L.jpg")
        photo_U_image.save(f"images/TrainData/Extraction/Photo_U/{num_extraction_images + 1}_U.jpg" if label == 1 else f"images/TrainData/Non_Extraction/Photo_U/{num_non_extraction_images + 1}_U.jpg")
        xray_image.save(f"images/TrainData/Extraction/Xray/{num_extraction_images + 1}_lat.jpg" if label == 1 else f"images/TrainData/Non_Extraction/Xray/{num_non_extraction_images + 1}_lat.jpg")
        num_extraction_images += 1 if label == 1 else 0
        num_non_extraction_images += 1 if label == 0 else 0

        return JSONResponse(content={"message": "Data uploaded successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/train")
async def train():
    try:
        # Define paths and labels
        photo_paths_L = []
        for i in range(1, num_extraction_images):
            photo_paths_L.append(f"images/TrainData/Extraction/Photo_L/{i}_L.jpg")
        for i in range(1, num_non_extraction_images):
            photo_paths_L.append(f"images/TrainData/Non_Extraction/Photo_L/{i}_L.jpg")
        photo_paths_U = []
        for i in range(1, num_extraction_images):
            photo_paths_U.append(f"images/TrainData/Extraction/Photo_U/{i}_U.jpg")
        for i in range(1, num_non_extraction_images):
            photo_paths_U.append(f"images/TrainData/Non_Extraction/Photo_U/{i}_U.jpg")

        xray_paths = []
        for i in range(1, num_extraction_images):
            xray_paths.append(f"images/TrainData/Extraction/Xray/{i}_lat.jpg")
        for i in range(1, num_non_extraction_images):
            xray_paths.append(f"images/TrainData/Non_Extraction/Xray/{i}_lat.jpg")

        labels = [1] * num_extraction_images + [0] * num_non_extraction_images

        # Create an instance of the dataset
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

        # Train the model
        train(model, train_loader, val_loader, device)
        return JSONResponse(content={"message": "Model trained successfully"})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
