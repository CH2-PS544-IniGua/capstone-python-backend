from fastapi import APIRouter, HTTPException, status
from app.models.user_model import User
from app.services.user_service import UserService
from typing import List

router = APIRouter()

@router.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
def create_user(user: User):
    created_user = UserService.create_user(user)
    return created_user

@router.get("/users/{username}", response_model=User)
def get_user(username: str):
    user = UserService.get_user(username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.get("/users", response_model=List[User])
def get_all_users():
    return UserService.get_all_users()