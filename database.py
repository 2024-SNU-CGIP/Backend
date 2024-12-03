from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from pathlib import Path

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    images = relationship("ImageMetadata", back_populates="patient")
    label = Column(Integer, index=True)
    timestamp = Column(String, index=True)

class ImageMetadata(Base):
    __tablename__ = "image_metadata"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'))
    photo_L_path = Column(String, index=True)
    photo_U_path = Column(String, index=True)
    xray_path = Column(String, index=True)
    patient = relationship("Patient", back_populates="images")

Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def db_init():
    db = SessionLocal()
    db.query(ImageMetadata).delete()
    db.query(Patient).delete()
    db.commit()
    db.close()

    for label in [0, 1]:
        photo_L_paths = sorted(Path(f"images/TrainData/{'Extraction' if label == 1 else 'Non_Extraction'}/Photo_L").glob("*.jpg"))
        photo_U_paths = sorted(Path(f"images/TrainData/{'Extraction' if label == 1 else 'Non_Extraction'}/Photo_U").glob("*.jpg"))
        xray_paths = sorted(Path(f"images/TrainData/{'Extraction' if label == 1 else 'Non_Extraction'}/Xray").glob("*.jpg"))
        
        for photo_L_path, photo_U_path, xray_path in zip(photo_L_paths, photo_U_paths, xray_paths):
            db = SessionLocal()
            patient = Patient(label=label, timestamp=str(datetime.now().timestamp()))
            db.add(patient)
            db.commit()
            db.refresh(patient)
            
            db.add(ImageMetadata(
                patient_id=patient.id,
                photo_L_path=str(photo_L_path) if photo_L_path else None,
                photo_U_path=str(photo_U_path) if photo_U_path else None,
                xray_path=str(xray_path) if xray_path else None
            ))
            db.commit()
            db.close()

if __name__ == "__main__":
    db_init()