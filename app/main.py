from fastapi import FastAPI
from app.api.user_routes import router as user_router
from app.api.fashion_routes import router as fashion_router

app = FastAPI()

app.include_router(user_router)
app.include_router(fashion_router)

# For running with `uvicorn` directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)