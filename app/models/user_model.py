from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class User(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    joined_at: Optional[datetime] = None
