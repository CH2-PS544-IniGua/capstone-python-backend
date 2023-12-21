from fastapi import APIRouter, File, Form, UploadFile, Depends
from app.services.fashion_service import FashionService
from app.models.fashion_model import FashionItem
import subprocess
import cv2
import json
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import argparse
import time
from pathlib import Path

import json
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import os
from app.api.models.experimental import attempt_load
from app.api.utils.datasets import LoadStreams, LoadImages
from app.api.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from app.api.utils.plots import plot_one_box
from app.api.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

router = APIRouter()

@router.post("/fashion")
async def upload_fashion(username: str = Form(...), picture: UploadFile = File(...), 
                         service: FashionService = Depends(FashionService)):
    content = await picture.read()  # Read file as bytes
    original_filename = picture.filename
    
    # Convert bytes to an image array
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the image with your ML model
    # Assuming that the `process_image` function is adjusted to accept an image array instead of a file path
    processed_image_result = process_image(img, original_filename)
    processed_image_info = processed_image_result[0]  # Assuming this is the format you're returning
    segmented_image = processed_image_info.get('segmented_image')
    print(processed_image_result)
    
    # Convert the processed image to the correct format for upload
    _, encoded_image = cv2.imencode('.jpg', segmented_image)
    byte_content = encoded_image.tobytes()
    
    # Create a FashionItem with the processed image
    fashion_item = FashionItem(username=username, picture=byte_content, filename=original_filename)
    
    # Upload the segmented image
    fashion_item_url = await service.upload_to_bucket(fashion_item)
    
    # Add the record to Firestore
    history_result = await service.add_to_firestore(
        username, 
        fashion_item.get_filename(), 
        fashion_item_url
    )
    
    return history_result

def detect_function(img_array, weights, name, img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False,
                    save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False,
                    augment=False, update=False, project='runs/detect', exist_ok=False, no_trace=False):

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    if no_trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Make sure the image array has 3 channels (RGB)
    if img_array.ndim == 2 or (img_array.ndim == 3 and img_array.shape[-1] != 3):
        # If it's a grayscale image (2D array), stack it three times to create 3 channels
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Convert the image array to a torch tensor
    img = torch.from_numpy(img_array).to(device)

    # If the image array was in HWC format, convert it to CHW format expected by PyTorch
    if img.ndim == 3 and img.shape[-1] == 3:
        img = img.permute(2, 0, 1)  # Convert HWC to CHW

    # Normalize and add batch dimension
    img = img.float() / 255.0  # normalize to 0 - 1 range
    img = img.unsqueeze(0)  # add batch dimension

    # Resize image to the input size expected by the model
    img = torch.nn.functional.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=False)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()
    
    # Inference
    t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()

    # Initialize result dictionary
    result = {
        "path": name,
        "prediction":[],
        "img": None  # This will be the image with drawn boxes
    }

    # Process detections
    for det in pred:  # detections for image
        im0 = img_array.copy()  # copy of original image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_img or view_img:  # Add bbox to image
                #     label = f'{names[int(cls)]} {conf:.2f}'
                #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                prediction_info = {
                        "class": int(cls),
                        "label": names[int(cls)],
                        "confidence": float(conf),
                        "bounding_box": [float(coord) for coord in xyxy],
                    }
                result["prediction"].append(prediction_info)
        result["img"] = im0  # Assign modified image to result dict

    # Print time (inference + NMS)
    print(f'Done. ({time.time() - t0:.3f}s)')
    return result

def detect(image_path, image_name):
    result = detect_function(image_path,'./best.pt', image_name)
    return result

def predict_and_display(image, model):
    img_array = preprocess_image(image)
    CATEGORIES = ['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return CATEGORIES[predicted_class]

def preprocess_image(image_array, target_size=(32,32)):
    resized_image = cv2.resize(image_array, (target_size[1], target_size[0]))
    img_array = np.expand_dims(resized_image, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def process_image(image_array, image_name='image'):
    # Assume the 'detect' function is now adapted to accept an image array and an image name
    segmentation_result = detect(image_array, image_name)

    # Use image_array directly instead of reading from the image_path
    image = image_array
    highest_confidence_per_label = {}
    highest_confidence_per_label = {}
    for prediction in segmentation_result['prediction']:
        bounding_box = prediction['bounding_box']
        label = prediction['label']
        confidence = prediction['confidence']

        bounding_box = [int(coord) for coord in bounding_box]
        if label not in highest_confidence_per_label or confidence > highest_confidence_per_label[label]['confidence']:
            highest_confidence_per_label[label] = {
                'bounding_box': bounding_box,
                'label': label,
                'confidence': confidence
            }
    
    result_prediction = []
    pred = {
            image_name:{
                
            }
            ,
            "segmented_image": segmentation_result["img"]
        }
    cropped_image =[]
    for label, highest_confidence_prediction in highest_confidence_per_label.items():
        bounding_box = highest_confidence_prediction['bounding_box']
        label = highest_confidence_prediction['label']
        confidence = highest_confidence_prediction['confidence']

        # Crop the region of interest (ROI) using the bounding box
        cropped_roi = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]

        # Save the cropped ROI to a file
        cropped_by_label = {}
        cropped_by_label[label] = cropped_roi
        cropped_image.append(cropped_by_label)
        color_model = load_model('./app/api/color_classification_cnn_model.h5')
        categories = predict_and_display(image, color_model)

        convert_putih = ["Pink","Cream","Gray","Red","Yellow"]
        convert_brown = ["Purple","Orange","Green","Blue"]
        if label == "skin":
            if categories in convert_putih:
                categories = "White"
            elif categories in convert_brown:
                categories = "Brown"

        pred[image_name][label] = categories
    result_prediction.append(pred)

    return result_prediction