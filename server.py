from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm import Query as SQLAlchemyQuery
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from pathlib import Path
import torchvision.transforms as transforms
from model import FusionModelMobileNetV2
from dataset import DentalDataset
from train import train, evaluate_model
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from database import SessionLocal, ImageMetadata, engine, get_db

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
    xray: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Save the images
        photo_L_image = Image.open(BytesIO(await photo_L.read()))
        photo_U_image = Image.open(BytesIO(await photo_U.read()))
        xray_image = Image.open(BytesIO(await xray.read()))

        photo_L_path = f"images/TrainData/Extraction/Photo_L/{photo_L.filename}" if label == 1 else f"images/TrainData/Non_Extraction/Photo_L/{photo_L.filename}"
        photo_U_path = f"images/TrainData/Extraction/Photo_U/{photo_U.filename}" if label == 1 else f"images/TrainData/Non_Extraction/Photo_U/{photo_U.filename}"
        xray_path = f"images/TrainData/Extraction/Xray/{xray.filename}" if label == 1 else f"images/TrainData/Non_Extraction/Xray/{xray.filename}"

        photo_L_image.save(photo_L_path)
        photo_U_image.save(photo_U_path)
        xray_image.save(xray_path)

        # Save metadata to the database
        db.add(ImageMetadata(filename=photo_L.filename, label=label, path=photo_L_path))
        db.add(ImageMetadata(filename=photo_U.filename, label=label, path=photo_U_path))
        db.add(ImageMetadata(filename=xray.filename, label=label, path=xray_path))
        db.commit()

        return JSONResponse(content={"message": "Data uploaded successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 상�� 정의
TEST_SIZE = 0.1
VAL_SIZE = 0.2
RANDOM_STATE = 42

def process_images(images, label, photo_paths_L, photo_paths_U, xray_paths):
    labels = []
    for image in images:
        if "Photo_L" in image.path:
            photo_paths_L.append(image.path)
        elif "Photo_U" in image.path:
            photo_paths_U.append(image.path)
        elif "Xray" in image.path:
            xray_paths.append(image.path)
        labels.append(label)
    return labels

@app.get("/train")
async def train_model(db: Session = Depends(get_db)):
    try:
        # Define paths and labels
        photo_paths_L = []
        photo_paths_U = []
        xray_paths = []
        labels = []

        extraction_images = db.query(ImageMetadata).filter(ImageMetadata.label == 1).all()
        non_extraction_images = db.query(ImageMetadata).filter(ImageMetadata.label == 0).all()

        labels.extend(process_images(extraction_images, 1, photo_paths_L, photo_paths_U, xray_paths))
        labels.extend(process_images(non_extraction_images, 0, photo_paths_L, photo_paths_U, xray_paths))

        # Split the data into train, validation, and test sets
        train_indices, test_indices = train_test_split(range(len(labels)), test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE)
        train_indices, val_indices = train_test_split(train_indices, test_size=VAL_SIZE, stratify=[labels[i] for i in train_indices], random_state=RANDOM_STATE)

        # Create dataset and dataloaders
        dataset = DentalDataset(photo_paths_L, photo_paths_U, xray_paths, labels, photo_transform, xray_transform)
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=32, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_indices), batch_size=32, shuffle=False)
        test_loader = DataLoader(Subset(dataset, test_indices), batch_size=32, shuffle=False)

        # Define loss function and optimizer
        criterion = lambda x, y: nn.BCELoss()(x, y) + 0.5 * nn.MSELoss()(x, y)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # Train the model
        train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=5)

        # evaluate the model
        test_accuracy = evaluate_model(model, test_loader, criterion, device)

        return JSONResponse(content={"message": "Model trained successfully", "test_accuracy": test_accuracy})
    except Exception as e:
        # 예외 발생 시 로그 추가
        print(f"Error during model training: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/images")
async def get_images(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1), db: Session = Depends(get_db)):
    try:
        offset = (page - 1) * page_size
        images_query: SQLAlchemyQuery = db.query(ImageMetadata).offset(offset).limit(page_size)
        images = images_query.all()
        total_images = db.query(ImageMetadata).count()
        return {
            "page": page,
            "page_size": page_size,
            "total_images": total_images,
            "images": images
        }
    except Exception as e:
        print(f"Error fetching images: {e}")
        raise HTTPException(status_code=500, detail="Error fetching images")

@app.get("/images/{image_id}")
async def get_image(image_id: int, db: Session = Depends(get_db)):
    try:
        image = db.query(ImageMetadata).filter(ImageMetadata.id == image_id).first()
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return image
    except Exception as e:
        print(f"Error fetching image: {e}")
        raise HTTPException(status_code=500, detail="Error fetching image")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
