from pydantic import BaseModel
from datetime import datetime

class FashionItem(BaseModel):
    username: str
    filename: str
    picture: bytes  # This will be the content of the picture file

    def filename(self):
        time_uploaded = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{self.filename}-{time_uploaded}"
    
    def file_username(self):
        return f"{self.username}"
