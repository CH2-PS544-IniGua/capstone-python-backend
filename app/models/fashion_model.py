from pydantic import BaseModel
from datetime import datetime
import os

class FashionItem(BaseModel):
    username: str
    filename: str
    picture: bytes  # This will be the content of the picture file

    def get_filename(self):
        # Separate the base name and the extension
        base, extension = os.path.splitext(self.filename)
        time_uploaded = datetime.now().strftime("%Y%m%d%H%M%S")
        # Insert the time_uploaded between the base name and the extension
        return f"{base}-{time_uploaded}{extension}"
    
    def get_username(self):
        return f"{self.username}"
