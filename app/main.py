import os
from fastapi import FastAPI
from app.api.fashion_routes import router as fashion_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.include_router(fashion_router)

# Get the port number from the environment variable or default to 8000
port = int(os.environ.get("PORT", 8080))
print(f"Running on port: {port}")

# For running with `uvicorn` directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=port)
