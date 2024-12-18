from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
from uuid import uuid4
from database import get_db, Train, Patient, ImageMetadata
from train import train, evaluate_model
from dataset import DentalDataset
from model import model, photo_transform, xray_transform, device
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim

router = APIRouter()

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

def train_model_task(db: Session, task_id: str):
    try:
        db.query(Train).filter(Train.id == task_id).update({"status": "in_progress"})
        db.commit()

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

        highest_accuracy = max(
            (result.test_accuracy for result in db.query(Train).all() if result.test_accuracy is not None),
            default=0
        )

        old_weights = torch.load('model_weights.pth', map_location=device)

        # Train the model
        train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, patience=5)

        # 학습 종료 시간 기록
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        # evaluate the model
        test_accuracy = evaluate_model(model, test_loader, device)

        if test_accuracy < highest_accuracy:
            torch.save(old_weights, 'model_weights.pth')
            model.load_state_dict(old_weights)

        db.query(Train).filter(Train.id == task_id).update({
            "status": "completed",
            "test_accuracy": test_accuracy,
            "training_time": training_time
        })
        db.commit()
    except Exception as e:
        db.query(Train).filter(Train.id == task_id).update({
            "status": "failed",
            "result": str(e)
        })
        db.commit()

@router.get("/train")
async def train_model(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    task_id = str(uuid4())  # 변경: task_id를 str로 변경
    db.add(Train(id=task_id, status="queued", timestamp=str(datetime.now().timestamp())))
    db.commit()
    background_tasks.add_task(train_model_task, db, task_id)
    return JSONResponse(content={"message": "Training started", "task_id": task_id})

@router.get("/train_result/{task_id}")
async def get_train_result(task_id: str, db: Session = Depends(get_db)):
    result = db.query(Train).filter(Train.id == task_id).first()
    if result:
        return JSONResponse(content={
            "status": result.status,
            "result": result.result,
            "test_accuracy": result.test_accuracy,
            "training_time": result.training_time
        })
    else:
        return JSONResponse(content={"message": "Task not found"}, status_code=404)

@router.get("/train_results")
async def get_all_train_results(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1), db: Session = Depends(get_db)):
    try:
        offset = (page - 1) * page_size
        tasks = db.query(Train).order_by(Train.timestamp.desc()).offset(offset).limit(page_size).all()
        total_tasks = db.query(Train).count()
        max_page = (total_tasks + page_size - 1) // page_size
        results = [
            {
            "id": task.id,
            "status": task.status,
            "test_accuracy": task.test_accuracy,
            "training_time": task.training_time
            } for task in tasks
        ]
        return {
            "page": page,
            "page_size": page_size,
            "total_tasks": total_tasks,
            "max_page": max_page,
            "results": results
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
