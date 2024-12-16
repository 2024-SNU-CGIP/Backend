from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
from uuid import uuid4
import os
from PIL import Image
from io import BytesIO
import base64
import cv2
from database import get_db, Predict, ImageMetadata
from visualizeXAI import guided_backpropagation, save_gradient, gen_circles
from model import model, photo_transform, xray_transform, device

router = APIRouter()

def predict_task(photo_L_path, photo_U_path, xray_path, task_id, db: Session):
    try:
        db.query(Predict).filter(Predict.id == task_id).update({"status": "in_progress"})
        db.commit()

        # Load and preprocess the images
        photo_L_image = Image.open(photo_L_path).convert('RGB')
        photo_U_image = Image.open(photo_U_path).convert('RGB')
        xray_image = Image.open(xray_path).convert('L')

        photo_L_tensor = photo_transform(photo_L_image).unsqueeze(0).requires_grad_(True).to(device)
        photo_U_tensor = photo_transform(photo_U_image).unsqueeze(0).requires_grad_(True).to(device)
        xray_tensor = xray_transform(xray_image).unsqueeze(0).requires_grad_(True).to(device)

        # Perform the prediction and guided backpropagation
        output, guided_grads_L, guided_grads_U, guided_grads_xray = guided_backpropagation(model, photo_L_tensor, photo_U_tensor, xray_tensor)

        # Save the gradients as images
        save_gradient(guided_grads_L, f"predict_images/{task_id}_grad_L.jpg")
        save_gradient(guided_grads_U, f"predict_images/{task_id}_grad_U.jpg")

        # Generate and save the XAI visualization for each image
        hitmap_L = cv2.imread(f"predict_images/{task_id}_grad_L.jpg", cv2.IMREAD_COLOR)
        image_with_circles_L = gen_circles(cv2.imread(photo_L_path), hitmap_L)
        cv2.imwrite(f"predict_images/{task_id}_xai_result_L.jpg", image_with_circles_L)

        hitmap_U = cv2.imread(f"predict_images/{task_id}_grad_U.jpg", cv2.IMREAD_COLOR)
        image_with_circles_U = gen_circles(cv2.imread(photo_U_path), hitmap_U)
        cv2.imwrite(f"predict_images/{task_id}_xai_result_U.jpg", image_with_circles_U)

        db.query(Predict).filter(Predict.id == task_id).update({"status": "completed", "result": str(output.item())})
        db.commit()
    except Exception as e:
        print(f"Prediction failed: {e}")
        db.query(Predict).filter(Predict.id == task_id).update({"status": "failed", "result": str(e)})
        db.commit()

@router.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    birthdate: str = Form(...),
    photo_L: UploadFile = File(...),
    photo_U: UploadFile = File(...),
    xray: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Ensure the directory exists
        os.makedirs("predict_images", exist_ok=True)

        # Save the images temporarily
        photo_L_path = f"predict_images/{uuid4()}.jpg"
        photo_U_path = f"predict_images/{uuid4()}.jpg"
        xray_path = f"predict_images/{uuid4()}.jpg"

        with open(photo_L_path, "wb") as f:
            f.write(await photo_L.read())
        with open(photo_U_path, "wb") as f:
            f.write(await photo_U.read())
        with open(xray_path, "wb") as f:
            f.write(await xray.read())

        task_id = str(uuid4())
        db.add(Predict(id=task_id, status="queued", timestamp=str(datetime.now().timestamp()), name=name, birthdate=birthdate, result=None))
        db.commit()
        background_tasks.add_task(predict_task, photo_L_path, photo_U_path, xray_path, task_id, db)
        return JSONResponse(content={"message": "Prediction started", "task_id": task_id})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/predict_result/{task_id}")
async def get_predict_result(task_id: str, db: Session = Depends(get_db)):
    result = db.query(Predict).filter(Predict.id == task_id).first()
    if result:
        image_data = {}
        if result.status == "queued":
            image_metadata = db.query(ImageMetadata).filter(ImageMetadata.patient_id == result.id).first()
            if image_metadata:
                if image_metadata.photo_L_path:
                    with open(image_metadata.photo_L_path, "rb") as photo_L_file:
                        image_data["photo_L"] = base64.b64encode(photo_L_file.read()).decode('utf-8')
                if image_metadata.photo_U_path:
                    with open(image_metadata.photo_U_path, "rb") as photo_U_file:
                        image_data["photo_U"] = base64.b64encode(photo_U_file.read()).decode('utf-8')
                if image_metadata.xray_path:
                    with open(image_metadata.xray_path, "rb") as xray_file:
                        image_data["xray"] = base64.b64encode(xray_file.read()).decode('utf-8')
        elif result.status == "completed":
            with open(f"predict_images/{task_id}_xai_result_L.jpg", "rb") as xai_result_L_file:
                image_data["photo_L"] = base64.b64encode(xai_result_L_file.read()).decode('utf-8')
            with open(f"predict_images/{task_id}_xai_result_U.jpg", "rb") as xai_result_U_file:
                image_data["photo_U"] = base64.b64encode(xai_result_U_file.read()).decode('utf-8')
            image_metadata = db.query(ImageMetadata).filter(ImageMetadata.patient_id == result.id).first()
            with open(image_metadata.xray_path, "rb") as xray_file:
                image_data["xray"] = base64.b64encode(xray_file.read()).decode('utf-8')
        return JSONResponse(content={"status": result.status, "result": result.result, "images": image_data})
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@router.get("/predict_results")
async def get_all_predict_results(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1), db: Session = Depends(get_db)):
    try:
        offset = (page - 1) * page_size
        tasks = db.query(Predict).order_by(Predict.timestamp.desc()).offset(offset).limit(page_size).all()
        total_tasks = db.query(Predict).count()
        max_page = (total_tasks + page_size - 1) // page_size
        results = [
            {
            "id": task.id,
            "status": task.status,
            "result": task.result,
            "name": task.name,
            "birthdate": task.birthdate
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
