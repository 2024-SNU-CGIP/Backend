from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session
from PIL import Image
from io import BytesIO
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.nn as nn
import torch.optim
from database import get_db, ImageMetadata, Patient
from dataset import DentalDataset  
from model import FusionModelMobileNetV2
from train import train, evaluate_model
import base64
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용, 특정 도메인만 허용하려면 ["http://example.com"]과 같이 설정
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

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
        photo_U_path = f"images/TrainData/Extraction/photo_U/{photo_U.filename}" if label == 1 else f"images/TrainData/Non_Extraction/photo_U/{photo_U.filename}"
        xray_path = f"images/TrainData/Extraction/Xray/{xray.filename}" if label == 1 else f"images/TrainData/Non_Extraction/Xray/{xray.filename}"

        photo_L_image.save(photo_L_path)
        photo_U_image.save(photo_U_path)
        xray_image.save(xray_path)

        # Save metadata to the database
        patient = Patient(label=label, timestamp=str(datetime.now().timestamp()))
        db.add(patient)
        db.commit()
        db.refresh(patient)

        db.add(ImageMetadata(
            patient_id=patient.id,
            photo_L_path=photo_L_path,
            photo_U_path=photo_U_path,
            xray_path=xray_path
        ))
        db.commit()

        return JSONResponse(content={"message": "Data uploaded successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 상수 정의
TEST_SIZE = 0.1
VAL_SIZE = 0.2
RANDOM_STATE = 42

def process_images(images, label, photo_paths_L, photo_paths_R, xray_paths):
    labels = []
    for image in images:
        if image.photo_L_path:
            photo_paths_L.append(image.photo_L_path)
        if image.photo_U_path:
            photo_paths_R.append(image.photo_U_path)
        if image.xray_path:
            xray_paths.append(image.xray_path)
        labels.append(label)
    return labels

training_results = {}
training_status = {}

def train_model_task(db: Session, task_id: str):
    try:
        training_status[task_id] = "in_progress"
        training_results[task_id] = {"message": "processing"}
        # Define paths and labels
        photo_paths_L = []
        photo_paths_R = []
        xray_paths = []
        labels = []

        patients = db.query(Patient).all()
        for patient in patients:
            images = db.query(ImageMetadata).filter(ImageMetadata.patient_id == patient.id).all()
            labels.extend(process_images(images, patient.label, photo_paths_L, photo_paths_R, xray_paths))

        # Split the data into train, validation, and test sets
        train_indices, test_indices = train_test_split(range(len(labels)), test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE)
        train_indices, val_indices = train_test_split(train_indices, test_size=VAL_SIZE, stratify=[labels[i] for i in train_indices], random_state=RANDOM_STATE)

        # Create dataset and dataloaders
        dataset = DentalDataset(photo_paths_L, photo_paths_R, xray_paths, labels, photo_transform, xray_transform)
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=32, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_indices), batch_size=32, shuffle=False)
        test_loader = DataLoader(Subset(dataset, test_indices), batch_size=32, shuffle=False)

        # Define loss function and optimizer
        criterion = lambda x, y: nn.BCELoss()(x, y) + 0.5 * nn.MSELoss()(x, y)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # 학습 시작 시간 기록
        start_time = datetime.now()

        # Train the model
        train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=5)

        # 학습 종료 시간 기록
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        # evaluate the model
        test_accuracy = evaluate_model(model, test_loader, criterion, device)

        training_results[task_id] = {"message": "Model trained successfully", "test_accuracy": test_accuracy, "training_time": training_time}
        training_status[task_id] = "completed"
    except Exception as e:
        training_status[task_id] = "failed"
        training_results[task_id] = {"error": str(e)}

@app.get("/train")
async def train_model(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    task_id = str(datetime.now().timestamp())
    total_patients_count = db.query(Patient).count()
    background_tasks.add_task(train_model_task, db, task_id)
    return JSONResponse(content={"message": "Training started", "task_id": task_id, "total_patients_count": total_patients_count})

@app.get("/train_result/{task_id}")
async def get_train_result(task_id: str):
    result = training_results.get(task_id)
    if result:
        return JSONResponse(content=result)
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.get("/train_results")
async def get_all_train_results():
    results = {task_id: {"status": status, **training_results.get(task_id, {})} for task_id, status in training_status.items()}
    return JSONResponse(content=results)

@app.get("/images")
async def get_images(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1), db: Session = Depends(get_db)):
    try:
        offset = (page - 1) * page_size
        patients_query = db.query(Patient).offset(offset).limit(page_size)
        patients = patients_query.all()
        total_patients = db.query(Patient).count()
        
        result = []
        for patient in patients:
            image = db.query(ImageMetadata).filter(ImageMetadata.patient_id == patient.id).first()
            result.append({
                "patient_id": patient.id,
                "label": patient.label,  # Include label
                "timestamp": datetime.fromtimestamp(float(patient.timestamp)).strftime('%Y:%m:%d %H:%M:%S'),  # Include timestamp
                "images": {
                    "id": image.id,
                    "photo_L_path": image.photo_L_path,
                    "photo_U_path": image.photo_U_path,
                    "xray_path": image.xray_path,
                }
            })
        
        return {
            "page": page,
            "page_size": page_size,
            "total_patients": total_patients,
            "patients": result
        }
    except Exception as e:
        print(f"Error fetching images: {e}")
        raise HTTPException(status_code=500, detail="Error fetching images")

@app.get("/images/{patient_id}")
async def get_image(patient_id: int, db: Session = Depends(get_db)):
    try:
        patient = db.get(Patient, patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        images = db.query(ImageMetadata).filter(ImageMetadata.patient_id == patient_id).all()
        image_data = []
        for image in images:
            image_info = {"id": image.id}
            if image.photo_L_path:
                with open(image.photo_L_path, "rb") as photo_L_file:
                    image_info["photo_L"] = base64.b64encode(photo_L_file.read()).decode('utf-8')
            if image.photo_U_path:
                with open(image.photo_U_path, "rb") as photo_U_file:
                    image_info["photo_U"] = base64.b64encode(photo_U_file.read()).decode('utf-8')
            if image.xray_path:
                with open(image.xray_path, "rb") as xray_file:
                    image_info["xray"] = base64.b64encode(xray_file.read()).decode('utf-8')
            image_data.append(image_info)
        
        return JSONResponse(content={
            "patient_id": patient.id,
            "label": patient.label,  # Include label
            "timestamp": datetime.fromtimestamp(float(patient.timestamp)).strftime('%Y:%m:%d %H:%M:%S'),  # Include timestamp
            "images": image_data
        })
    except Exception as e:
        print(f"Error fetching images: {e}")
        raise HTTPException(status_code=500, detail="Error fetching images")

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    try:
        # 최근 데이터 개수
        recent_data_count = db.query(ImageMetadata).count()

        # 최근 학습 날짜
        recent_train_date = max(training_results.keys(), default=None)
        if recent_train_date:
            recent_train_date = datetime.fromtimestamp(float(recent_train_date)).strftime('%Y:%m:%d %H:%M:%S')

        # 가장 높은 정확도
        highest_accuracy = max((result.get("test_accuracy", 0) for result in training_results.values()), default=0)

        return JSONResponse(content={
            "recent_data_count": recent_data_count,
            "recent_train_date": recent_train_date,
            "highest_accuracy": highest_accuracy
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
