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
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing_extensions import Annotated
from typing import Union


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/ontology_detection")
async def predict_ontology():

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

        print(ontology_satisfying_diseases)

    else:
        print("Failed to fetch ontology information")

    return {
        "disease": ontology_satisfying_diseases,
    }


@app.post("/extra_symptoms")
async def predict(
    payload: Dict[Any, Any]
):
    global EXTRA_SYMPTOMS
    EXTRA_SYMPTOMS = payload
    return{
        "message": "Data received successfully"
    }


@app.post("/extra_symptomss")
async def predictttt(payload: Dict[Any, Any]):
    print(payload)
    return {"message": "Data received successfully"}


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


# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)


# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    user = authenticate_user(
        fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    return current_user


@app.get("/users/me/items/")
async def read_own_items(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    return [{"item_id": "Foo", "owner": current_user.username}]


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
