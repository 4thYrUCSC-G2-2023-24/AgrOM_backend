import pytest
from fastapi.testclient import TestClient
# Replace 'your_main_module' with the name of your main FastAPI module
from main import app
import json

# Create a test client using TestClient
client = TestClient(app)


image_data = open(
    "C:\\Users\\USER\\Desktop\\Test Images\\Late_blight.JPG", "rb").read()


def test_ping():
    # Test the ping endpoint
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.text.strip('"') == "Hello, I am alive"


def test_predict_endpoint_with_extra_symptoms():

    response = client.post(
        "/image_upload",
        files={"file": ("Late_blight.JPG", image_data)},
    )
    assert response.status_code == 200
    assert "disease" in response.json()


def test_extra_symptoms_endpoint():
    # Define the payload data
    payload = {
        "hasLeafSymptom": "lesions",
        "hasLeafSymptomColour": "yellow",
    }

    # Send a POST request to the endpoint with the payload
    response = client.post("/extra_symptoms", json=payload)

    # Check if the response status code is 200
    assert response.status_code == 200

    # Check if the response message indicates that the data was received successfully
    assert response.json()["message"] == "Data received successfully"


def test_ontology_detection_endpoint():

    response = client.get(
        "/ontology_detection",
    )

    assert response.status_code == 200
    assert "disease" in response.json()
