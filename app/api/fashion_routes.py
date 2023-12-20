from fastapi import APIRouter, File, Form, UploadFile, Depends
from app.services.fashion_service import FashionService
from app.models.fashion_model import FashionItem
import json
import base64
import os

router = APIRouter()

@router.post("/fashion")
async def upload_fashion(username: str = Form(...), picture: UploadFile = File(...), 
                         service: FashionService = Depends(FashionService)):
    content = await picture.read()
    original_filename = picture.filename
    fashion_item = FashionItem(username=username, picture=content, filename=original_filename)
    await service.upload_to_bucket(fashion_item)
    return {"filename": fashion_item.filename()}
