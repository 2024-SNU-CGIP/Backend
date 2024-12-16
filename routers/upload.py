from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
from PIL import Image
from io import BytesIO
from database import get_db, Patient, ImageMetadata

router = APIRouter()

@router.post("/upload_data")
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
