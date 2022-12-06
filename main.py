# Import libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd 
import joblib


# get api
app = FastAPI()

# Models
class Applicant(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str 
    ApplicantIncome: int
    CoapplicantIncome: float 
    LoanAmount: float 
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


# load joblib
model_lr = joblib.load("model_lr.joblib")
model_rf = joblib.load("model_rf.joblib")


# Routes For testing connection
@app.get("/")
def home():
    return{"message": "Hello World"}


# For logistic regression
@app.post("/predict_lr")
def predict(data: Applicant):
    input_data = data.dict()

    # convert dict to pd dataframe
    header = []
    value = []
    for key, val in input_data.items():
        header.append(key)
        value.append(val)
    
    df = pd.DataFrame([value], columns=header)

    prediction = model_lr.predict(df)
    result = "Yes" if (prediction == 1) else "No"
    return {"result": result}


# For random_forest
@app.post("/predict_rf")
def predict(data: Applicant):
    input_data = data.dict()

    # convert dict to pd dataframe
    header = []
    value = []
    for key, val in input_data.items():
        header.append(key)
        value.append(val)
    
    df = pd.DataFrame([value], columns=header)

    # predict data
    prediction = model_rf.predict(df)
    result = "Yes" if (prediction == 1) else "No"
    return {"result": result}