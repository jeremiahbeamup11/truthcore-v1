from fastapi import FastAPI

app = FastAPI(title="TruthCore v1")

@app.get("/")
def read_root():
    return {
        "message": "TruthCore v1 API is running! 🚀",
        "status": "healthy",
        "version": "1.0"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}