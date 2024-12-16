from fastapi import FastAPI
from routers import predict, train, upload, images, stats, test
from middleware import add_cors_middleware

app = FastAPI()

add_cors_middleware(app)

app.include_router(predict.router, prefix="/predict")
app.include_router(train.router, prefix="/train")
app.include_router(upload.router, prefix="/upload")
app.include_router(images.router, prefix="/images")
app.include_router(stats.router, prefix="/stats")
app.include_router(test.router, prefix="/test")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
