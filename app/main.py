from fastapi import FastAPI
from app.api.user_routes import router as user_router

app = FastAPI()

app.include_router(user_router)

# For running with `uvicorn` directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)