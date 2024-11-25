from sqlalchemy import create_engine, Column, Integer, String, Float
from pathlib import Path
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ImageMetadata(Base):
    __tablename__ = "image_metadata"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    label = Column(Integer)
    path = Column(String)

Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def db_init():
    # insert all data to db iterating over all file in images folder
    db = SessionLocal()
    db.query(ImageMetadata).delete()
    db.commit()
    db.close()
    for label in [0, 1]:
        for image_type in ["Photo_L", "Photo_U", "Xray"]:
            for image_path in Path(f"images/TrainData/{'Extraction' if label == 1 else 'Non_Extraction'}/{image_type}").glob("*.jpg"):
                db = SessionLocal()
                db.add(ImageMetadata(filename=image_path.name, label=label, path=str(image_path)))
                db.commit()
                db.close()

if __name__ == "__main__":
    db_init()