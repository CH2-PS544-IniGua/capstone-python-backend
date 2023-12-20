import os
from fastapi import FastAPI
from app.api.user_routes import router as user_router
from app.api.fashion_routes import router as fashion_router

app = FastAPI()

app.include_router(user_router)
app.include_router(fashion_router)

# Get the port number from the environment variable or default to 8000
port = int(os.environ.get("PORT", 8000))

# For running with `uvicorn` directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
