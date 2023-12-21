from fastapi import APIRouter, File, Form, UploadFile, Depends
from app.services.fashion_service import FashionService
from app.models.fashion_model import FashionItem
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tempfile
import shutil

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
    file_extension = os.path.splitext(picture.filename)[1]

    # Save the content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name
        print(temp_file_path)

    try:
        # Process the image with your ML model (implementation not shown here)
        processed_image_result = process_image(temp_file_path, original_filename)
        processed_image_info = processed_image_result[0]
        segmented_image = processed_image_info.get('segmented_image')

        # Convert the processed image to the correct format for upload
        _, encoded_image = cv2.imencode('.jpg', segmented_image)
        byte_content = encoded_image.tobytes()

        # Create a FashionItem with the processed image
        fashion_item = FashionItem(username=username, picture=byte_content, filename=original_filename)


        # Upload the segmented image
        fashion_item_url = await service.upload_to_bucket(fashion_item)

        # UNTUK FIRESTORE
        labels = processed_image_info.get(original_filename)
        skin = labels.get('skin')
        body1 = labels.get('body1')
        body2 = labels.get('body2')

        # handle gaada
        skin = 'None' if not skin else skin
        body1 = 'None' if not body1 else body1
        body2 = 'None' if not body2 else body2

        fp_skin_body= 'app/api/percentage_skin_body.json'
        fp_body1_body2 = 'app/api/percentage_body1_body2.json'

        # Open the file and load its contents into a dictionary
        with open(fp_skin_body, 'r') as file:
            data_skin_body = json.load(file)

        with open(fp_body1_body2, 'r') as file:
            data_body1_body2 = json.load(file)

        # Add the record to Firestore (assuming this function is defined correctly in your service)
        history_result = await service.add_to_firestore(
            username, 
            fashion_item.get_filename(), 
            fashion_item_url,
            skin,
            body1,
            body2,
            0 if body1 == "None" or skin == "None" else data_skin_body[skin][body1],
            0 if body1 == "None" or body2 == "None" else data_body1_body2[body1][body2],
        )
    finally:
        # No matter what happens, make sure to clean up the temp file
        os.unlink(temp_file_path)

    return history_result

def detect_function(source, weights, name,img_size=640, conf_thres=0.25, iou_thres=0.45, device='', view_img=False,
                    save_txt=False, save_conf=False, nosave=False, classes=None, agnostic_nms=False,
                    augment=False, update=False, project='runs/detect',  exist_ok=False, no_trace=False):
    opt_source = source
    opt_weights = weights
    opt_img_size = img_size
    opt_conf_thres = conf_thres
    opt_iou_thres = iou_thres
    opt_device = device
    opt_view_img = view_img
    opt_save_txt = save_txt
    opt_save_conf = save_conf
    opt_nosave = nosave
    opt_classes = classes
    opt_agnostic_nms = agnostic_nms
    opt_augment = augment
    opt_update = update
    opt_project = project
    opt_name = name
    opt_exist_ok = exist_ok
    opt_no_trace = no_trace
    source, weights, view_img, save_txt, imgsz, trace = opt_source, opt_weights, opt_view_img, opt_save_txt, opt_img_size, not opt_no_trace
    # source, weights, view_img, save_txt, imgsz, trace = opt_source, opt_weights, opt_view_img, opt_save_txt, opt_img_size, not opt_no_trace
    save_img = not opt_nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # image_basename = os.path.splitext(os.path.basename(source))[0]
    # save_dir = Path(Path(opt_project) / opt_name, exist_ok=opt_exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # save_dir = Path(opt_project) / opt_name / "result" / image_basename
    # (save_dir / 'labels' if opt_save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt_device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt_img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt_augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt_augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres, classes=opt_classes, agnostic=opt_agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                result = {
                # "directory": str(save_dir),
                "path": str(source),
                "prediction":[],
                
                }
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
              
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt_save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                    prediction_info = {
                            "class": int(cls),
                            "label": names[int(cls)],
                            "confidence": float(conf),
                            "bounding_box": [float(coord) for coord in xyxy],
                            # "img": im0
                        }
                    result["img"] = im0
                    result["prediction"].append(prediction_info)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.time() - t0:.3f}s)')
    return result

def detect(image_path, image_name):
    result = detect_function(image_path,'./app/api/best.pt', image_name)
    return result
    # detection_command = f"python ./yolov7/detect.py --weights ./yolov7/best.pt --conf 0.1 --source {image_path} --name {image_name}"
    # result = subprocess.run(detection_command, shell=True, text=True, capture_output=True)

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

def process_image(image_path, image_name="image-name"):
    segmentation_result = detect(image_path, image_name)

    image_segmented_path = segmentation_result['path']


    image = cv2.imread(image_path)
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
            "path": image_segmented_path,
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
        rgb_image = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2RGB)
        cropped_image.append(cropped_by_label)
        color_model = load_model('./app/api/color_classification_cnn_model.h5')
        categories = predict_and_display(rgb_image, color_model)

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