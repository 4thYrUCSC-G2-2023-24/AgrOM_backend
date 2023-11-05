from fastapi import FastAPI, File, UploadFile, Body
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pydantic import BaseModel
from rdflib import Graph, Namespace
import requests


app = FastAPI()

MODEL = tf.keras.models.load_model("C:\\Users\\USER\\Desktop\\vgg model")
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
    #hasFruitSymptom: str = 'Spot_symptom'
    # hasFruitSymptomColor: str = None
    # hasStemSymptom: str = None
    # hasStemSymptomColor: str = None
    # hasLeafHalo: str = None
    # hasLeafHaloColor: str = None
    # hasFruitHalo: str = None
    # hasFruitHaloColor: str = None
    # hasFungusSymptom: str = None
    # hasFungusSymptomColor: str = None


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


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data, target_size=(224, 224)) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Resize the image
    resized_image = tf.image.resize(np.array(image), target_size)
    print(resized_image.shape)
    return resized_image


@app.post("/image_upload")
async def predict(
    file: UploadFile = File(...),
):
    image = await file.read()
    image = Image.open(BytesIO(image))
    resized_image = tf.image.resize(np.array(image), (224, 224))
    img_batch = np.expand_dims(resized_image, 0)
    predictions = MODEL.predict(img_batch)
    confidence_threshold = 0.0
    class_confidence_map = dict(zip(CLASS_NAMES, predictions[0]))
    filtered_diseases = [disease for disease,
                         confidence in class_confidence_map.items() if confidence >= confidence_threshold]
    Deep_Model_Sorted_Disease = sorted(
        filtered_diseases, key=lambda x: class_confidence_map[x], reverse=True)

    github_raw_uri = "https://raw.githubusercontent.com/mtbstn24/OntoML/main/OntoMLv3.rdf"

    # Fetch the RDF file from the GitHub repository
    response = requests.get(github_raw_uri)

    if response.status_code == 200:
        # Create a Graph
        g = Graph()
        g.parse(data=response.text, format="application/rdf+xml")

        query = build_sparql_query(EXTRA_SYMPTOMS)

        # Execute the SPARQL query
        results = g.query(query)

        ontology_satisfying_diseases = []

        for row in results:
            disease = row.diseaseName
            ontology_satisfying_diseases.append(disease.value)

        print("Ontology disease")

        print(ontology_satisfying_diseases)

        found = False
        output_disease = "Not found"
        if len(ontology_satisfying_diseases) == 0:
            output_disease = Deep_Model_Sorted_Disease[0]
        elif len(ontology_satisfying_diseases) == 1:
            output_disease = ontology_satisfying_diseases[0]
        elif len(ontology_satisfying_diseases) == 0 and len(Deep_Model_Sorted_Disease) == 0:
            print("Couldn't find the disease with given information")
        else:

            for deep_disease in Deep_Model_Sorted_Disease:
                for onto_disease in ontology_satisfying_diseases:
                    print(deep_disease)
                    print(onto_disease)
                    if disease_mapping[deep_disease] == onto_disease:
                        output_disease = deep_disease
                        found = True
                        break
                if found:
                    break

    else:
        print("Failed to fetch ontology information")

    return {
        "disease": output_disease,
    }


@app.post("/extra_symptoms")
async def predict(
    extra_symptoms: ExtraSymptomModelOut = Body(...)
):
    global EXTRA_SYMPTOMS
    EXTRA_SYMPTOMS = extra_symptoms.model_dump()
    return{
        "Extra symptoms received"
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
):
    image = await file.read()
    image = Image.open(BytesIO(image))
    resized_image = tf.image.resize(np.array(image), (224, 224))
    img_batch = np.expand_dims(resized_image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    return{
        "disease": predicted_class,
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
