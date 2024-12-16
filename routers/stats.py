from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
import json
from database import get_db, Train, ImageMetadata

router = APIRouter()

@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    try:
        # 최근 데이터 개수
        recent_data_count = db.query(ImageMetadata).count()

        # 최근 학습 날짜
        recent_train = db.query(Train).order_by(Train.timestamp.desc()).first()
        recent_train_date = datetime.fromtimestamp(float(recent_train.timestamp)).strftime('%Y:%m:%d %H:%M:%S') if recent_train else None

        # 가장 높은 정확도
        highest_accuracy = max(
            (result.test_accuracy for result in db.query(Train).all() if result.test_accuracy is not None),
            default=0
        )

        return JSONResponse(content={
            "recent_data_count": recent_data_count,
            "recent_train_date": recent_train_date,
            "highest_accuracy": highest_accuracy
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
