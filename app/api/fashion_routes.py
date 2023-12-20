from fastapi import APIRouter, File, Form, UploadFile, Depends
from app.services.fashion_service import FashionService
from app.models.fashion_model import FashionItem

router = APIRouter()

@router.post("/fashion")
async def upload_fashion(username: str = Form(...), picture: UploadFile = File(...), 
                         service: FashionService = Depends(FashionService)):
    content = await picture.read()
    original_filename = picture.filename
    fashion_item = FashionItem(username=username, picture=content, filename=original_filename)
    fashion_item_url = await service.upload_to_bucket(fashion_item)
    history_result = await service.add_to_firestore(username, fashion_item.get_filename(), fashion_item_url)
    return history_result
