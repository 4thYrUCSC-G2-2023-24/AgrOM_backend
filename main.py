import base64
from fastapi import FastAPI, File, UploadFile, Body
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pydantic import BaseModel
from rdflib import Graph, Namespace
import requests
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing_extensions import Annotated
from typing import Union
from motor.motor_asyncio import AsyncIOMotorClient
import json
import cv2
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt
import pickle
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Construct the path to your model file
model_path_vgg16 = os.path.join(parent_dir, "saved_models", "vgg16_model")
model_path_yolov8_lsd = os.path.join(parent_dir, "saved_models", "yolov8lsd_model", "best.pt")
model_path_yolov8_dld = os.path.join(parent_dir, "saved_models", "yolov8dld_model", "best.pt")
model_path_sam = os.path.join(parent_dir, "saved_models", "sam_model", "sam_vit_h_4b8939.pth")
model_path_yolov8_dls = os.path.join(parent_dir, "saved_models", "yolov8dls_model", "best.pt")
model_ontology = os.path.join(parent_dir, "ontology_file", "OntoMLv3.rdf")

MODEL = tf.keras.models.load_model(
    model_path_vgg16
)

MODEL_VGG16 = tf.keras.models.load_model(model_path_vgg16)
MODEL_YOLOV8_DLD = YOLO(model_path_yolov8_dld)
MODEL_YOLOV8_LSD = YOLO(model_path_yolov8_lsd)
MODEL_SAM = sam_model_registry["vit_h"](checkpoint=model_path_sam)

class ImageData(BaseModel):
    image_base64: str

CLASS_NAMES = ['Tomato_Bacterial_spot',
               'Tomato_Early_blight',
               'Tomato_Late_blight',
               'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Black_mold',
               'Tomato__Gray_spot',
               'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus',
               'Tomato__powdery_mildew',
               'Tomato_healthy']

disease_mapping = {
    "Tomato_Bacterial_spot": "bacterial_spot",
    "Tomato_Early_blight": "early_blight",
    "Tomato_Late_blight": "late_blight",
    "Tomato_Leaf_Mold": "leaf_mold",
    "Tomato_Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "spider_mites",
    "Tomato__Black_mold": "black_mold",
    "Tomato__Gray_spot": "gray_spot",
    "Tomato__Target_Spot": "target_spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "yellow_leaf_curl",
    "Tomato__Tomato_mosaic_virus": "mosaic_virus",
    "Tomato__powdery_mildew": "powdery_mildew",
    "Tomato_healthy": "healthy"
}


class ExtraSymptomModelOut(BaseModel):
    hasLeafSymptom: str = 'lesions'
    hasLeafSymptomColour: str = 'yellow'
    hasFruitSymptom: str = 'Spot_symptom'
    hasFruitSymptomColor: str = None
    hasStemSymptom: str = None
    hasStemSymptomColor: str = None
    hasLeafHalo: str = None
    hasLeafHaloColor: str = None
    hasFruitHalo: str = None
    hasFruitHaloColor: str = None
    hasFungusSymptom: str = None
    hasFungusSymptomColor: str = None


def build_sparql_query(symptoms):
    # Build SPARQL query based on symptoms
    query = """PREFIX OntoML: <https://github.com/mtbstn24/OntoMLv3#>
SELECT DISTINCT (strafter(STR(?disease), "#") AS ?diseaseName)
WHERE {
?disease rdf:type OntoML:Disease.
"""
    for symptom, value in symptoms.items():
        if value:
            query += f"?disease OntoML:{symptom} OntoML:{value}.\n"
    query += "}"
    return query


def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(laplacian_var)
    return laplacian_var < threshold


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data, target_size=(224, 224)) -> np.ndarray:
    image = Image.open(BytesIO(data))

    # Resize the image
    resized_image = tf.image.resize(np.array(image), target_size)
    print(resized_image.shape)
    return resized_image


class PredictionFacade:
    def __init__(self, model_path, ontology_url):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
                            'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
                            'Tomato__Black_mold', 'Tomato__Gray_spot', 'Tomato__Target_Spot',
                            'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                            'Tomato__powdery_mildew', 'Tomato_healthy']
        self.disease_mapping = {
            "Tomato_Bacterial_spot": "bacterial_spot",
            "Tomato_Early_blight": "early_blight",
            "Tomato_Late_blight": "late_blight",
            "Tomato_Leaf_Mold": "leaf_mold",
            "Tomato_Septoria_leaf_spot": "septoria_leaf_spot",
            "Tomato_Spider_mites_Two_spotted_spider_mite": "spider_mites",
            "Tomato__Black_mold": "black_mold",
            "Tomato__Gray_spot": "gray_spot",
            "Tomato__Target_Spot": "target_spot",
            "Tomato__Tomato_YellowLeaf__Curl_Virus": "yellow_leaf_curl",
            "Tomato__Tomato_mosaic_virus": "mosaic_virus",
            "Tomato__powdery_mildew": "powdery_mildew",
            "Tomato_healthy": "healthy"
        }
        self.ontology_url = model_ontology

    def process_image(self, file):
        image = Image.open(BytesIO(file))
        resized_image = tf.image.resize(np.array(image), (224, 224))

        image_np = np.array(resized_image)
        # if len(image_np.shape) > 2 and image_np.shape[2] == 4:
        #     # convert the image from RGBA2RGB
        #     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
        # print(image_np.shape)
        image_np_uint8 = image_np.astype(np.uint8)

        if self.is_blurry(image_np_uint8):
            return {"disease": "Image is Blurry"}

        img_batch = np.expand_dims(resized_image, 0)
        predictions = self.model.predict(img_batch)
        print(predictions)
        confidence_threshold = 0.0
        class_confidence_map = dict(zip(self.class_names, predictions[0]))
        filtered_diseases = [disease for disease,
                             confidence in class_confidence_map.items() if confidence >= confidence_threshold]
        deep_model_sorted_disease = sorted(
            filtered_diseases, key=lambda x: class_confidence_map[x], reverse=True)

        ontology_satisfying_diseases = self.query_ontology()

        output_disease = self.match_disease(
            deep_model_sorted_disease, ontology_satisfying_diseases)

        return {"disease": output_disease}

    def is_blurry(self, image, threshold=50):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold

    # def query_ontology(self):
    #     response = requests.get(self.ontology_url)
    #     if response.status_code == 200:
    #         g = Graph()
    #         g.parse(data=response.text, format="application/rdf+xml")
    #         results = g.query(build_sparql_query(EXTRA_SYMPTOMS))
    #         ontology_satisfying_diseases = [
    #             row.diseaseName.value for row in results]
    #         return ontology_satisfying_diseases
    #     else:
    #         raise Exception("Failed to fetch ontology information")

    def query_ontology(self):
        # response = requests.get(self.ontology_url)
        # if response.status_code == 200:
        #     g = Graph()
        #     g.parse(data=response.text, format="application/rdf+xml")
        #     results = g.query(build_sparql_query(EXTRA_SYMPTOMS))
        #     ontology_satisfying_diseases = [
        #         row.diseaseName.value for row in results]
        #     return ontology_satisfying_diseases
        # else:
        #     raise Exception("Failed to fetch ontology information")

        g = Graph()
        g.parse(model_ontology,
                format="application/rdf+xml")
        results = g.query(build_sparql_query(EXTRA_SYMPTOMS))
        ontology_satisfying_diseases = [
            row.diseaseName.value for row in results]
        return ontology_satisfying_diseases

    def match_disease(self, deep_model_sorted_disease, ontology_satisfying_diseases):
        output_disease = "Not found"
        if not deep_model_sorted_disease:
            return output_disease

        if len(ontology_satisfying_diseases) == 0:
            output_disease = deep_model_sorted_disease[0]
        elif len(ontology_satisfying_diseases) == 1:
            output_disease = ontology_satisfying_diseases[0]
        else:
            for deep_disease in deep_model_sorted_disease:
                for onto_disease in ontology_satisfying_diseases:
                    if self.disease_mapping.get(deep_disease) == onto_disease:
                        output_disease = deep_disease
                        return output_disease
        return output_disease

    def detect_ontology(self):
        # response = requests.get(self.ontology_url)
        # if response.status_code == 200:
        #     g = Graph()
        #     g.parse(data=response.text, format="application/rdf+xml")
        #     results = g.query(build_sparql_query(EXTRA_SYMPTOMS))
        #     ontology_satisfying_diseases = [
        #         row.diseaseName.value for row in results]
        #     return {"disease": ontology_satisfying_diseases}
        # else:
        #     raise Exception("Failed to fetch ontology information")
        g = Graph()
        g.parse(model_ontology,
                format="application/rdf+xml")
        results = g.query(build_sparql_query(EXTRA_SYMPTOMS))
        ontology_satisfying_diseases = [
            row.diseaseName.value for row in results]
        return {"disease": ontology_satisfying_diseases}


# C:\\Users\\USER\\Desktop\\vgg model

@app.post("/image_upload")
async def predict(file: UploadFile = File(...)):
    facade = PredictionFacade(model_path_vgg16,
                              "https://raw.githubusercontent.com/Sribarathvajasarma/Plant_disease_ontology_2/main/OntoMLv3.rdf")
    result = facade.process_image(await file.read())
    return result


# https://raw.githubusercontent.com/mtbstn24/OntoML/main/OntoMLv3.rdf


@app.get("/ontology_detection")
async def predict_ontology():
    facade = PredictionFacade(model_path_vgg16,
                              "https://raw.githubusercontent.com/Sribarathvajasarma/Plant_disease_ontology_2/main/OntoMLv3.rdf")
    result = facade.detect_ontology()
    return result


# @app.post("/image_upload")
# async def predict(
#     file: UploadFile = File(...),
# ):

#     image = await file.read()
#     image = Image.open(BytesIO(image))
#     resized_image = tf.image.resize(np.array(image), (224, 224))

#     image_np = np.array(resized_image)
#     image_np_uint8 = image_np.astype(np.uint8)

#     # EXTRA_SYMPTOMS = {
#     #     "hasLeafSymptom": "lesions",
#     #     "hasLeafSymptomColour": "yellow",
#     # }

#     if is_blurry(image_np_uint8):
#         return {
#             "error": "Image is Blurry",
#         }

#     img_batch = np.expand_dims(resized_image, 0)
#     predictions = MODEL.predict(img_batch)
#     confidence_threshold = 0.0
#     class_confidence_map = dict(zip(CLASS_NAMES, predictions[0]))
#     filtered_diseases = [disease for disease,
#                          confidence in class_confidence_map.items() if confidence >= confidence_threshold]
#     Deep_Model_Sorted_Disease = sorted(
#         filtered_diseases, key=lambda x: class_confidence_map[x], reverse=True)

#     github_raw_uri = "https://raw.githubusercontent.com/mtbstn24/OntoML/main/OntoMLv3.rdf"

#     # Fetch the RDF file from the GitHub repository
#     response = requests.get(github_raw_uri)

#     if response.status_code == 200:
#         # Create a Graph
#         g = Graph()
#         g.parse(data=response.text, format="application/rdf+xml")

#         query = build_sparql_query(EXTRA_SYMPTOMS)

#         # Execute the SPARQL query
#         results = g.query(query)

#         ontology_satisfying_diseases = []

#         for row in results:
#             disease = row.diseaseName
#             ontology_satisfying_diseases.append(disease.value)

#         print("Ontology disease")

#         print(ontology_satisfying_diseases)

#         found = False
#         output_disease = "Not found"
#         if len(ontology_satisfying_diseases) == 0:
#             output_disease = Deep_Model_Sorted_Disease[0]
#         elif len(ontology_satisfying_diseases) == 1:
#             output_disease = ontology_satisfying_diseases[0]
#         elif len(ontology_satisfying_diseases) == 0 and len(Deep_Model_Sorted_Disease) == 0:
#             print("Couldn't find the disease with given information")
#         else:

#             for deep_disease in Deep_Model_Sorted_Disease:
#                 for onto_disease in ontology_satisfying_diseases:
#                     print(deep_disease)
#                     print(onto_disease)
#                     if disease_mapping[deep_disease] == onto_disease:
#                         output_disease = deep_disease
#                         found = True
#                         break
#                 if found:
#                     break

#     else:
#         print("Failed to fetch ontology information")

#     return {
#         "disease": output_disease,
#     }


# @app.get("/ontology_detection")
# async def predict_ontology():

#     # payload = {
#     #     "hasLeafSymptom": "lesions",
#     #     "hasLeafSymptomColour": "yellow",
#     # }

#     github_raw_uri = "https://raw.githubusercontent.com/mtbstn24/OntoML/main/OntoMLv3.rdf"

#     # Fetch the RDF file from the GitHub repository
#     response = requests.get(github_raw_uri)

#     if response.status_code == 200:
#         # Create a Graph
#         g = Graph()
#         g.parse(data=response.text, format="application/rdf+xml")

#         query = build_sparql_query(EXTRA_SYMPTOMS)

#         # Execute the SPARQL query
#         results = g.query(query)

#         ontology_satisfying_diseases = []

#         for row in results:
#             disease = row.diseaseName
#             ontology_satisfying_diseases.append(disease.value)

#         print(ontology_satisfying_diseases)

#     else:
#         print("Failed to fetch ontology information")

#     return {
#         "disease": ontology_satisfying_diseases,
#     }


@app.post("/extra_symptoms")
async def predict(
    payload: Dict[Any, Any]
):
    global EXTRA_SYMPTOMS
    EXTRA_SYMPTOMS = payload
    print(EXTRA_SYMPTOMS)
    return{
        "message": "Data received successfully"
    }


@app.post("/extra_symptomss")
async def predictttt(payload: Dict[Any, Any]):
    print(payload)
    return {"message": "Data received successfully"}


def decode_base64_image(image_base64: str):
    try:
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

@app.post("/predict_disease")
async def predict_disease(
    file: UploadFile = File(...),
):
    image = await file.read()
    image = Image.open(BytesIO(image))
    resized_image = tf.image.resize(np.array(image), (224, 224))
    img_batch = np.expand_dims(resized_image, 0)
    predictions = MODEL_VGG16.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    return{
        "disease": predicted_class,
    }

@app.post("/predict_disease_test")
async def predict_disease(
    image_data: ImageData,
):
    image = decode_base64_image(image_data.image_base64)
    resized_image = tf.image.resize(np.array(image), (224, 224))
    img_batch = np.expand_dims(resized_image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    return{
        "disease": predicted_class,
    }

def apply_bounding_boxes(img, results, names):
    detected = []
    for result in results:
        annotator = Annotator(np.ascontiguousarray(img))
        
        boxes = result.boxes
        for box in boxes:
            # get box coordinates in (left, top, right, bottom) format
            b = box.xyxy[0]  
            c = box.cls
            prob = round(box.conf[0].item(), 2)
            print(prob)
            # annotator.box_label(b, names[int(c)] + ": " + str(prob))
            annotator.box_label(b, names[int(c)])
            detected.append(names[int(c)])
    img = annotator.result()
    return img, detected

# Convert NumPy array to base64 string
def numpy_array_to_base64(array):
    _, buffer = cv2.imencode('.png', array)  # Convert to PNG format
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/upload_process_image")
async def detect_tomato_leaves(file: UploadFile = File(...)):
    image = await file.read()
    image = Image.open(BytesIO(image))

    results = MODEL_YOLOV8_DLD.predict(source=image, task='detect', conf=0.4, save=False)
    boundingbox_image, leaves_detected = apply_bounding_boxes(image, results, MODEL_YOLOV8_DLD.names)
    cv2.imwrite("./detect.png", boundingbox_image) 

    base64_image = numpy_array_to_base64(boundingbox_image)

    return{
        "boundingbox_image": base64_image,
        "leaves_detected": len(leaves_detected),
    }

@app.post("/upload_detect_symptoms")
async def detect_leaves_symptoms(file: UploadFile = File(...)):
    image = await file.read()
    image = Image.open(BytesIO(image))
    
    results = MODEL_YOLOV8_LSD.predict(source=image, task='detect', conf=0.35, save=False)
    boundingbox_image, symptoms_detected = apply_bounding_boxes(image, results, MODEL_YOLOV8_LSD.names)
    if(len(symptoms_detected) == 0):
        results = MODEL_YOLOV8_LSD.predict(source=image, task='detect', conf=0.2, save=False)
        boundingbox_image, symptoms_detected = apply_bounding_boxes(image, results, MODEL_YOLOV8_LSD.names)
    
    cv2.imwrite("./detect.png", boundingbox_image)

    # Remove underscores from strings and get unique items
    unique_symptoms = list(set(item.replace('_', ' ') for item in symptoms_detected))
    
    base64_image = numpy_array_to_base64(boundingbox_image)

    return{
        "boundingbox_image": base64_image,
        "symptoms_detected": unique_symptoms,
    }

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def apply_segmentation_mask(img, results, names):
    import matplotlib
    matplotlib.use('agg')

    # Get the width and height of the image
    width, height = img.size
    
    img = np.array(img)
    masks_list = []

    for result in results:
        boxes = result.boxes

        plt.figure(figsize=(width / 100, height / 100))
        plt.imshow(img)
        
        for box in boxes:
            cls = box.cls
            prob = round(box.conf[0].item(), 2)
            class_name = names[int(cls)]
            print(prob)
            

            if len(cls) > 0:
                # get box coordinates in (left, top, right, bottom) format
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                
                # # Ploat the rectangle around the detected object
                # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # # text to the rectangle
                # text = class_name
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 1.5
                # thickness = 4
                # text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                # text_x = int(x1 + 5)
                # text_y = int(y1 + text_size[1] + 5)
                # cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

                # SAM Model
                predictor = SamPredictor(MODEL_SAM)
                predictor.set_image(img)

                input_box = np.array(box.xyxy.tolist()[0])
                print("input_box", input_box)

                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )

                masks_list.append(masks[0].astype(np.uint8))
                
                show_mask(masks[0], plt.gca())
                show_box(input_box, plt.gca())

        plt.axis('off')

        # Get the current figure as a NumPy array
        fig = plt.gcf()
        fig.canvas.draw()
        mask_img = np.array(fig.canvas.renderer.buffer_rgba())

        plt.savefig('mask.png')
        plt.close()
    

    seg_img = segment_leaves(img, masks_list)

    return mask_img, seg_img

def segment_leaves(image, masks_list):
    combined_mask = np.zeros_like(masks_list[0], dtype=np.uint8)
    for mask in masks_list:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Apply the combined mask to the original image
    segmented_image = cv2.bitwise_and(image, image, mask=combined_mask)

    # Display or save the segmented image
    cv2.imwrite("segment.png", segmented_image)

    return segmented_image

def alt_apply_segmentation_mask(img, results, names):
    for result in results:
        annotator = Annotator(np.ascontiguousarray(img))
        
        for box in result.boxes:
            b = box.xyxy[0]  
            c = box.cls
            prob = round(box.conf[0].item(), 2)
            print(prob)
            # annotator.box_label(b, names[int(c)] + ": " + str(prob))
            annotator.box_label(b, names[int(c)])
        
        for mask in result.masks:
            cv2.addWeighted(img, 1, mask, 0.5, 0)

    img = annotator.result()
    return img


@app.post("/upload_segment_leaves")
async def segment_leaf_images(file: UploadFile = File(...)):
    image = await file.read()
    image = Image.open(BytesIO(image))
    
    results = MODEL_YOLOV8_DLD.predict(source=image, task='detect', conf=0.45, save=False)
    try:
        mask_img, seg_img = apply_segmentation_mask(image, results, MODEL_YOLOV8_DLD.names)
    except:
        results = MODEL_YOLOV8_DLD.predict(source=image, task='detect', conf=0.1, save=False)
        mask_img, seg_img = apply_segmentation_mask(image, results, MODEL_YOLOV8_DLD.names)

    base64_mask_image = numpy_array_to_base64(mask_img)
    base64_seg_image = numpy_array_to_base64(seg_img)

    return{
        "mask_image": base64_mask_image,
        "segment_image": base64_seg_image,
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
