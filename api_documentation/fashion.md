# Fashion Service API Documentation

This document details the Fashion Service API, which allows users to upload fashion items and add them to their history.

## Endpoints

### Upload Fashion Item and Update History

`POST /fashion`

Allows a user to upload a fashion item image, which is then stored in a bucket and its details are added to the user's history.

- **Form Data Parameters:**
  - `username`: The username of the user uploading the fashion item.
  - `picture`: The image file of the fashion item to be uploaded.

- **Response:**

  ```json
  {
    "filename": "<string>",
    "predict_image": "<image URL>",
    "datetime": "<datetime string>",
    "color_bottom": "<string>",
    "color_skin": "<string>",
    "color_top": "<string>",
    "percentage_clothes_pants": <integer>,
    "percentage_skin_clothes": <integer>
  }
