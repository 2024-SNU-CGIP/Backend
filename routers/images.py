from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
import base64
from database import get_db, Patient, ImageMetadata

router = APIRouter()

@router.get("/images")
async def get_images(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1), db: Session = Depends(get_db)):
    try:
        offset = (page - 1) * page_size
        patients_query = db.query(Patient).offset(offset).limit(page_size)
        patients = patients_query.all()
        total_patients = db.query(Patient).count()
        max_page = (total_patients + page_size - 1) // page_size
        
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
            "max_page": max_page,
            "patients": result
        }
    except Exception as e:
        print(f"Error fetching images: {e}")
        raise HTTPException(status_code=500, detail="Error fetching images")

@router.get("/images/{patient_id}")
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