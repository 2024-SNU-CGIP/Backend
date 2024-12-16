from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
from uuid import uuid4
from database import get_db, Train, Predict

router = APIRouter()

@router.post("/add_dummy_train_data")
async def add_dummy_train_data(db: Session = Depends(get_db)):
    try:
        # Add dummy train tasks
        for i in range(5):
            task_id = str(uuid4())
            db.add(Train(
                id=task_id,
                status="completed",
                result=str({"test_accuracy": 0.8 + i * 0.02, "training_time": 3600 + i * 100}),
                timestamp=str(datetime.now().timestamp())
            ))
            db.commit()

        return JSONResponse(content={"message": "Dummy train data added successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/add_dummy_predict_data")
async def add_dummy_predict_data(db: Session = Depends(get_db)):
    try:
        # Add dummy predict tasks
        for i in range(5):
            task_id = str(uuid4())
            db.add(Predict(
                id=task_id,
                status="completed",
                result=str(i%2),
                timestamp=str(datetime.now().timestamp()),
                name=f"Dummy Patient {i}",
                birthdate="2000-01-01"
            ))
            db.commit()

        return JSONResponse(content={"message": "Dummy predict data added successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)