import time
import os
import warnings
import logging
import json
import torch
from torchvision import transforms
from PIL import Image
import io
import psutil
import cv2
from contextlib import contextmanager, redirect_stdout
from ultralytics import YOLO
from pathlib import Path

# Suppress all warnings
warnings.simplefilter("ignore")
# Disable oneDNN custom operations warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress Ultralytics logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Memory usage function
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

# Leaf classes
leaf_classes_1 = [
    'Apple', 'Apple', 'Apple', 'Apple', 'Apple', 'Cherry', 'Cherry',
    'Corn', 'Corn', 'Corn', 'Corn', 'Grape', 'Grape', 'Grape', 'Grape',
    'Orange', 'Sugarcane', 'Sugarcane', 'corn', 'Strawberry', 'Potato', 'Potato',
    'Potato', 'Potato', 'Soybean', 'Squash', 'Tomato', 'Tomato',
    'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato', 'Tomato',
    'Tomato', 'Tomato', 'Tomato'
]

# Load TorchScript ResNet-9 model
leaf_model_path = 'plant_model16.ptl'
leaf_model = torch.jit.load(leaf_model_path, map_location=torch.device('cpu'))
leaf_model.eval()

# ResNet-9 Prediction Function
# ResNet-9 Prediction Function
def predict_resnet9(image_path, model=leaf_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    try:
        initial_memory = get_memory_usage()
        start_time = time.time()
        with open(image_path, 'rb') as f:
            image = Image.open(io.BytesIO(f.read()))
            img_t = transform(image)
            img_u = torch.unsqueeze(img_t, 0).to(dtype=torch.float16)  # Cast to float16
             # Move the input tensor to GPU


        with torch.autocast(dtype=torch.float16,device_type='cpu'):  # Ensure model runs in float16
            yb = model(img_u)
            probs = torch.nn.functional.softmax(yb, dim=1)
            confidence, preds = torch.max(probs, dim=1)
        
        time_taken = time.time() - start_time
        memory_usage = get_memory_usage() - initial_memory

        prediction = leaf_classes_1[preds[0].item()]
        confidence_score = round(confidence[0].item() * 100, 2)

        return [str(prediction), str(confidence_score), str(round(time_taken, 2)), str(round(memory_usage, 2))]
    except Exception as e:
        return [str(f"Error: {str(e)}"), "0.0", "0.0", "0.0"]


# YOLO Model Prediction Function
def processing_result_box(result):
    try:
        result_string = str(result)
        cleaned_result = result_string[7:-1]
        nested_list = json.loads(cleaned_result)
        class_names = [
            'ginger', 'banana', 'tobacco', 'ornamental', 'rose', 'soyabean',
            'papaya', 'garlic', 'raspberry', 'mango', 'cotton', 'corn', 'pomegranate',
            'strawberry', 'blueberry', 'brinjal', 'potato', 'wheat', 'olive',
            'rice', 'lemon', 'cabbage', 'guava', 'chilli', 'capsicum',
            'sunflower', 'cherry', 'cassava', 'apple', 'tea', 'sugarcane',
            'groundnut', 'weed', 'peach', 'coffee', 'cauliflower', 'tomato',
            'onion', 'gram', 'chiku', 'jamun', 'castor', 'pea', 'cucumber',
            'grape', 'cardamom'
        ]
        name_of_plant = class_names[int(nested_list[0][5])]
        confidence = round(nested_list[0][4] * 100, 2)
        return [str(name_of_plant), str(confidence)]
    except (json.JSONDecodeError, IndexError):
        return ['Unknown Plant', "0.0"]

def predict_yolo(image_path):
    try:
        initial_memory = get_memory_usage()
        start_time = time.time()
        model = YOLO('best.pt')
        image = cv2.imread(image_path)
        if image is None:
            return ["Error: Image not found or invalid format", "0.0", "0.0", "0.0"]

        results = model(image)
        time_taken = time.time() - start_time
        ans = processing_result_box(results[0].boxes)
        memory_usage = get_memory_usage() - initial_memory

        return [ans[0], ans[1], str(round(time_taken, 2)), str(round(memory_usage, 2))]
    except Exception as e:
        return [str(f"Error: {str(e)}"), "0.0", "0.0", "0.0"]

# Example Usage
if __name__ == "__main__":
    image_path = 'tomato.jpg'
    if Path(image_path).is_file():
        print(f"Image is :{image_path}")
       

        yolo_result = predict_yolo(image_path)
        print("YOLO Result:", yolo_result)
        resnet_result = predict_resnet9(image_path)
        print("ResNet-9 Result:", resnet_result)
    else:
        print(f"Error: Image file '{image_path}' not found.")
