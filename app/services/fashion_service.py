import base64
import os
import json
from datetime import datetime
from google.cloud import storage, firestore
from google.oauth2 import service_account
from app.models.fashion_model import FashionItem
from dotenv import load_dotenv

load_dotenv()

class FashionService:
    def __init__(self):
        service_account_info = json.loads(base64.b64decode(os.environ.get('SERVICE_ACCOUNT')))
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        self.storage_client = storage.Client(credentials=credentials)
        self.firestore_client = firestore.Client(credentials=credentials)
        self.bucket_name = 'application-inigua'

    async def upload_to_bucket(self, fashion_item: FashionItem):
        bucket = self.storage_client.get_bucket(self.bucket_name)
        fashion_folder_path = f"fashion/{fashion_item.get_username()}/{fashion_item.get_filename()}"
        blob = bucket.blob(fashion_folder_path)
        blob.upload_from_string(fashion_item.picture, content_type='image/jpeg')  # Adjust the content_type if needed
        blob.make_public()  # Make the blob publicly viewable
        return blob.public_url

    async def add_to_firestore(self, username, filename, imgurl):
        # Create a reference to the Firestore collection
        history_ref = self.firestore_client.collection('history').document(username).collection('history')
        history_data = {
            'filename': filename,
            'predict_image': imgurl,
            'datetime': datetime.now().isoformat(),  # Store the current time in ISO format
            'color_bottom': 'Green',
            'color_skin': 'Brown',
            'color_top': 'Pink',
            'percentage_clothes_pants': 37,
            'percentage_skin_clothes': 69
        }
        # Add the history record to Firestore
        history_ref.add(history_data)
