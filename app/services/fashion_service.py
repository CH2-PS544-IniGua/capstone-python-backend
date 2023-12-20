import base64
import os
import json
from google.cloud import storage
from google.oauth2 import service_account
from app.models.fashion_model import FashionItem
from dotenv import load_dotenv

load_dotenv()

class FashionService:
    def __init__(self):
        service_account_info = json.loads(base64.b64decode(os.environ.get('SERVICE_ACCOUNT')))
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        self.client = storage.Client(credentials=credentials)
        self.bucket_name = 'application-inigua'

    async def upload_to_bucket(self, fashion_item: FashionItem):
        bucket = self.client.get_bucket(self.bucket_name)
        fashion_folder_path = f"fashion/{fashion_item.get_username()}/{fashion_item.get_filename()}"
        blob = bucket.blob(fashion_folder_path)
        blob.upload_from_string(fashion_item.picture, content_type='image/jpeg')  # Adjust the content_type if needed

        # Make the blob publicly viewable (if necessary)
        blob.make_public()
        return blob.public_url
