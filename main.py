from sqlalchemy import create_engine
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import pandas as pd
from config import ACCESS_TOKEN_EXPIRE_MINUTES
from database_conn import database
from helper import authenticate_user, create_access_token, get_current_active_user
from custom_data_type import Token, User, spend_analysis_input, trend_analysis_input, supplier_evaluation_input, demand_forecasting_input
# from azure.storage.blob.baseblobservice import BaseBlobService
# from azure.storage.blob import BlobPermissions
from datetime import datetime, timedelta
from azure.storage.blob import ContainerClient
from io import StringIO
import json
import numpy as np
from io import BytesIO
from spend_analysis_function import spend_analysis
from trend_analysis_function import port_item_count_port
from supplier_evaluation_function import supplier_evaluation
from demand_forecast_function import Demand_forecast
import re
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# from nltk.stem import PorterStemmer
# from Levenshtein import seqratio
# from nltk.tokenize import word_tokenize
# from functools import reduce
# import re
# connection
container_client = ContainerClient.from_connection_string(
    'DefaultEndpointsProtocol=https;AccountName=treewiseblobstorage;AccountKey=jE3f/ogf+EH2cZyJEEagULdbWrIFvtKOnJB655pvrSn+9jzniIx8hGjHlBvnb3Py2I6h7b5zO2NO+AStfk0NPA==;EndpointSuffix=core.windows.net', container_name='maersk-vendor-recommendation-db')


lead_time_weight = 0.35
price_weight = 0.35
rating_weight = 0.3


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def database_connect():
    await database.connect()


@app.on_event("shutdown")
async def database_disconnect():
    await database.disconnect()


# Authentication
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
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


@app.post("/spend_analysis")
async def fetch_data(userinput: spend_analysis_input, current_user: User = Depends(get_current_active_user)):
    item_cat = userinput.item_cat
    item_sec1 = userinput.item_sec1
    item_sec2 = userinput.item_sec2
    item = userinput.item
    data = read_data_from_blob("IMPA_ITEMS_WITH_PORT.csv")
    c = spend_analysis(data, item_cat, item_sec1, item_sec2, item)
    result_dict = c.to_dict(orient='records')
    return {'data': result_dict}


@app.post("/trend_analysis")
async def fetch_data(userinput: trend_analysis_input, current_user: User = Depends(get_current_active_user)):
    item_cat = userinput.item_cat
    item_sec1 = userinput.item_sec1
    item_sec2 = userinput.item_sec2
    port = userinput.port
    data = read_data_from_blob("IMPA_ITEMS_WITH_PORT.csv")
    c = port_item_count_port(data, item_cat, item_sec1, item_sec2, portt=port)
    result_dict = c.to_dict(orient='records')
    return {'data': result_dict}


@app.post("/supplier_evaluation")
async def fetch_data(userinput: supplier_evaluation_input, current_user: User = Depends(get_current_active_user)):
    vendor = userinput.vendor
    df = read_data_from_blob("IMPA_ITEMS_WITH_PORT.csv")
    c = supplier_evaluation(df, vendor)
    result_dict = c.to_dict(orient='records')
    return {'data': result_dict}


@app.post("/demand_forecasting")
async def fetch_data(userinput: demand_forecasting_input, current_user: User = Depends(get_current_active_user)):
    vessel_type = userinput.vessel_type
    vessel_sub_type= userinput.vessel_sub_type
    df = read_data_from_blob("3.Demand_Forecasting_Items_Victualling_12_01_2023.csv")
    c = Demand_forecast(df, vessel_type, vessel_sub_type)
    # print(c)
    # result_dict = c.to_dict(orient='records')
    return {'data': c}


# def read_data_from_blob(dataset_name):
#     try:
#         content = container_client.download_blob(
#             dataset_name).content_as_text(encoding='latin-1')
#         data = pd.read_csv(StringIO(content))
#         return data
#     except Exception as e:
#         print("Exception in reading from BLOB", e)
#         raise HTTPException(
#             status_code=404, detail='Error in reading data from blob storage')
def read_data_from_blob(dataset_name):
    try:
        data = pd.read_csv(StringIO(container_client.download_blob(dataset_name).content_as_text()))
        return data
    except Exception as e:
        print("EXcception in reading from BLOB",e)
        raise HTTPException(status_code=404, detail='Error in reading data from blob storage')
