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


class ExtraSymptomModelOut(BaseModel):
    hasLeafSymptom: str = 'Lesion_symptom'
    # hasLeafSymptomColor: str = None
    hasFruitSymptom: str = 'Spot_symptom'
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
    query = """
        SELECT ?disease
        WHERE {
    """
    for symptom, value in symptoms.items():
        if value:
            query += f"?disease ontology_ns:{symptom} ontology_ns:{value}.\n"
    query += "}"

    print(query)
    return query


@app.get("/ping")
async def ping():
    return "Hello, I am aliveeee"


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
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence_threshold = 0.0
    confidence = np.max(predictions[0])
    class_confidence_map = dict(zip(CLASS_NAMES, predictions[0]))
    filtered_diseases = [disease for disease,
                         confidence in class_confidence_map.items() if confidence >= confidence_threshold]
    Deep_Model_Sorted_Disease = sorted(
        filtered_diseases, key=lambda x: class_confidence_map[x], reverse=True)

    # Deep_Model_Sorted_Disease = sorted(
    #     CLASS_NAMES, key=lambda x: class_confidence_map[x], reverse=True)
    # print(predictions[0])
    print(Deep_Model_Sorted_Disease)

    github_raw_uri = "https://raw.githubusercontent.com/Sribarathvajasarma/Plant_Disease_Ontology/main/Plant_Disease_ontology.owl"

    # Fetch the RDF file from the GitHub repository
    response = requests.get(github_raw_uri)

    if response.status_code == 200:
        # Create a Graph
        g = Graph()
        g.parse(data=response.text, format="application/rdf+xml")

        ontology_ns = Namespace(
            "https://raw.githubusercontent.com/Sribarathvajasarma/Plant_Disease_Ontology/main/Plant_Disease_ontology.owl#")
        rdf_ns = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

        # SPARQL query to extract diseases and their symptoms
        # query = """
        #     SELECT ?disease
        #     WHERE {
        #         ?disease ontology_ns:hasFruitSymptom ontology_ns:Spot_symptom.
        #     }
        # """

        query = build_sparql_query(EXTRA_SYMPTOMS)

        # Execute the SPARQL query
        results = g.query(
            query, initNs={"ontology_ns": ontology_ns, "rdf": rdf_ns})

        ontology_satisfying_diseases = []

        # Extract and print the results
        for row in results:
            disease_uri = row.disease
            disease_name = disease_uri.split("#")[-1]
            ontology_satisfying_diseases.append(disease_name)

        output_disease = []
        # selection algorithm
        if len(ontology_satisfying_diseases) == 0:
            output_disease.append(Deep_Model_Sorted_Disease[0])
        elif (len(ontology_satisfying_diseases) != 0 and Deep_Model_Sorted_Disease[0]):
            output_disease = ontology_satisfying_diseases
        elif (len(ontology_satisfying_diseases) == 0 and Deep_Model_Sorted_Disease[0] == 0):
            return {
                "Couldn't predict the disease using given information"
            }
        else:
            for deep_disease in Deep_Model_Sorted_Disease:
                for onto_disease in Deep_Model_Sorted_Disease:
                    if deep_disease == onto_disease:
                        output_disease.append(deep_disease)

    else:
        print("Failed to fetch ontology information")

    return {
        "disease": predicted_class,
        "confidence": float(confidence),
        "Ontology_Disease": ontology_satisfying_diseases,
        "output_disease": output_disease
    }


@app.post("/extra_symptoms")
async def predict(
    extra_symptoms: ExtraSymptomModelOut = Body(...)
):
    global EXTRA_SYMPTOMS
    EXTRA_SYMPTOMS = extra_symptoms.model_dump()
    #query = build_sparql_query(EXTRA_SYMPTOMS)
    return{
        "Extra symptoms received"
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
