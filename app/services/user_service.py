from app.models.user_model import User
from app.database import users_db
from typing import List

class UserService:
    @staticmethod
    def create_user(user: User) -> User:
        if user.username in users_db:
            raise ValueError("Username already registered")
        users_db[user.username] = user
        return user

    @staticmethod
    def get_user(username: str) -> User:
        return users_db.get(username)

    def get_all_users() -> List[User]:
        return [data for data in users_db.values()]
