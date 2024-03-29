from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import pandas as pd
from config import ACCESS_TOKEN_EXPIRE_MINUTES
from database_conn import database
from helper import authenticate_user, create_access_token, get_current_active_user
from custom_data_type import Token, User, spend_analysis_input, trend_analysis_input,supplier_evaluation_input,demand_forecasting_input
# from azure.storage.blob.baseblobservice import BaseBlobService
# from azure.storage.blob import BlobPermissions
from datetime import datetime, timedelta
from azure.storage.blob import ContainerClient
from io import StringIO
import json
import numpy as np
import tempfile
from io import BytesIO
from spend_analysis_function import spend_analysis
from preprocessing_predictor import predict_overall_score
from trend_analysis_function import port_item_count_port
from supplier_evaluation_function import supplier_evaluation
from demand_forecast_function import Demand_forecast
import re
import tensorflow as tf
import joblib
import pickle
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# from nltk.stem import PorterStemmer
# from Levenshtein import seqratio
# from nltk.tokenize import word_tokenize
# from functools import reduce
# import re
#connection
container_client = ContainerClient.from_connection_string(
    'DefaultEndpointsProtocol=https;AccountName=treewiseblobstorage;AccountKey=jE3f/ogf+EH2cZyJEEagULdbWrIFvtKOnJB655pvrSn+9jzniIx8hGjHlBvnb3Py2I6h7b5zO2NO+AStfk0NPA==;EndpointSuffix=core.windows.net', container_name='maersk-vendor-recommendation-db')


from sqlalchemy import create_engine

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
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.post("/spend_analysis")
async def fetch_data(current_user: User = Depends(get_current_active_user)):
    data = read_data_from_blob("3.Demand_Forecasting_Items_Victualling_12_01_2023.csv")
    c = spend_analysis(data)
    result_dict = c.to_dict(orient='records')
    return {'data': result_dict}
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.post("/trend_analysis")
async def fetch_data(userinput: trend_analysis_input , current_user: User = Depends(get_current_active_user)):
    item_cat = userinput.item_cat
    item_sec1 = userinput.item_sec1
    item_sec2 = userinput.item_sec2
    port = userinput.port
    data = read_data_from_blob("IMPA_ITEMS_WITH_PORT.csv")
    c =port_item_count_port(data,item_cat,item_sec1,item_sec2,portt=port)
    result_dict = c.to_dict(orient='records')
    return {'data': result_dict}
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.post("/supplier_evaluation")

async def fetch_data(userinput: supplier_evaluation_input , current_user: User = Depends(get_current_active_user)):
    item = userinput.item
    port = userinput.port
    po_qty= userinput.po_qty
    df = read_data_from_blob("FINAL_OUTPUT_PREDICTOR.csv")
    print("data read")
    model =  tf.keras.models.load_model("Supplier_Evaluation.h5")
    print("model loaded")
    preprocessor=joblib.load('preprocessor.joblib')
    print("preprocessor loaded")
    c =predict_overall_score(df,item,port,model,preprocessor,po_qty)
    result_dict = c.to_dict(orient='records')
    return {'data': result_dict}
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# @app.post("/demand_forecasting_dev")
# async def fetch_data(userinput: demand_forecasting_input, current_user: User = Depends(get_current_active_user)):
#     vessel_type = userinput.vessel_type
#     vessel_sub_type= userinput.vessel_sub_type
#     df = read_data_from_blob("3.Demand_Forecasting_Items_Victualling_12_01_2023.csv")
#     c = Demand_forecast(df, vessel_type, vessel_sub_type)
#     # print(c)
#     # result_dict = c.to_dict(orient='records')
#     return {'data': c}

@app.post("/demand_forecasting")
async def fetch_data(userinput: demand_forecasting_input, current_user: User = Depends(get_current_active_user)):
    vessel_name = userinput.vessel_name
    n_qtrs = userinput.number_quarters
    if n_qtrs in [1,2,3,4]:
    # Calculate the next 4 quarters and their corresponding years
        current_date = datetime.now()
        year_quarters = []
        for i in range(0, 4): #(1,5)
            # Calculate the start date of the next quarter
            months_to_add = 3 * i
            future_date = current_date + timedelta(days=months_to_add * 30)  # Approximation
            future_quarter = (future_date.month - 1) // 3 + 1
            year_quarters.append(str(future_date.year)+'Q'+str(future_quarter))
            # year_quarters.append(str(future_date.year))

        container_client = ContainerClient.from_connection_string(
                'DefaultEndpointsProtocol=https;AccountName=treewiseblobstorage;AccountKey=jE3f/ogf+EH2cZyJEEagULdbWrIFvtKOnJB655pvrSn+9jzniIx8hGjHlBvnb3Py2I6h7b5zO2NO+AStfk0NPA==;EndpointSuffix=core.windows.net', container_name='maersk-vendor-recommendation-db')
        vessel_info = read_data_from_blob('1.Maersk_Demo_Data_Vessel_Type_Sub_Type_Details.csv')
        vsl_data = vessel_info[vessel_info['Vessel']==vessel_name]

        if len(vsl_data)>0:
            vessel_type,vessel_sub_type = str(list(vsl_data['BSM_VESSEL_TYPE'])[0]), str(list(vsl_data['BSM_VESSEL_SUBTYPE'])[0])
            # vessel_type = userinput.vessel_type
            # vessel_sub_type= userinput.vessel_sub_type
            with open('type_ids.pickle','rb') as file11:
                type_ids = pickle.load(file11)
            vsl_comb = [keys for keys in type_ids.keys() if vessel_type.lower().strip()+'_'+vessel_sub_type.lower().strip() in  keys.lower()]
            if len(vsl_comb)>0:    
                path_endpoint_list = list(container_client.list_blobs('Demand_Forecast_res/'))
                path_endpoint_list = [blobs['name'].split('/')[-1] for blobs in path_endpoint_list]
                
                filtered_list = []
                for pp in path_endpoint_list:
                    if pp.split('_')[0] == str(type_ids[str(vessel_type)+'_'+str(vessel_sub_type)]):
                        if pp.split('_')[1].split('.')[0] in year_quarters:
                            filtered_list.append(pp)
                if len(filtered_list)!=0:    
                    with open('UoMs.pickle','rb') as um:
                        UoM = pickle.load(um)    
                    filtered_list.sort()  
                    qtr_end_point_result = {}
                    for ids in range(n_qtrs):
                        blob_client = container_client.get_blob_client('Demand_Forecast_res/'+filtered_list[0])
                        pickled_data = blob_client.download_blob().readall()
                        end_point_result = pickle.loads(pickled_data)     
                        end_point_result['SEA CHEF PROVISIONS'] = {k1:v1 for k1,v1 in end_point_result['SEA CHEF PROVISIONS'].items() if v1>=1}
                        end_point_result['PROVISION'] = {k2:v2 for k2,v2 in end_point_result['PROVISION'].items() if v2>=1}
                        # end_point_result['SEA CHEF PROVISIONS'] = {k1:str(v1)+' '+UoM['SEA CHEF PROVISIONS'][k1] for k1,v1 in end_point_result['SEA CHEF PROVISIONS'].items()}
                        end_point_result['SEA CHEF PROVISIONS'] = {k1:{'Qty':v1,'Unit':UoM['SEA CHEF PROVISIONS'][k1]} for k1,v1 in end_point_result['SEA CHEF PROVISIONS'].items()}
                        end_point_result['PROVISION'] = {k2:{'Qty':v2,'Unit':UoM['PROVISION'][k2]} for k2,v2 in end_point_result['PROVISION'].items()}
                        qtr_end_point_result[filtered_list[ids].split('.')[0].split('_')[1]] = end_point_result
                        print('keys -',qtr_end_point_result.keys())
                        del qtr_end_point_result[filtered_list[ids].split('.')[0].split('_')[1]]['VESSEL_TYPE']
                        del qtr_end_point_result[filtered_list[ids].split('.')[0].split('_')[1]]['VESSEL_SUB_TYPE']
                    qtr_end_point_result['VESSEL_TYPE'] = vessel_type
                    qtr_end_point_result['VESSEL_SUB_TYPE'] = vessel_sub_type
                    qtr_end_point_result['vessel_name'] = vessel_name
                else:
                    qtr_end_point_result = {'Inputs are incorrect':'Check vessel type or vessel sub type'}
                return qtr_end_point_result
            else:
                return {'Vessel type-sub type combinations are not matching':'Check vessel type or vessel sub type'}
        else:
            return {'Vessel data is not available':'Check vessel name'}
    else: 
        return {'Quarter value error':'Invalid quarter value'}   

def load_model_from_blob(model_name):
    try:
        # Downloading the blob data
        blob_data = container_client.download_blob(model_name)
        stream = BytesIO()
        blob_data.readinto(stream)
        stream.seek(0) 
        # Write the stream content into a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file_name = tmp_file.name
            tmp_file.write(stream.getbuffer())
            
        # Loading the model from the temporary file
        model = tf.keras.models.load_model(tmp_file_name)
        return model

    except Exception as e:
        print("Exception in reading model from BLOB", e)
        raise HTTPException(status_code=404, detail='Error in reading model from blob storage')
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_preprocessor_from_blob(preprocessor_name):
    try:
        blob_data = container_client.download_blob(preprocessor_name)
        stream = BytesIO()
        blob_data.readinto(stream)
        
        # Load the preprocessor from the stream
        stream.seek(0)  # Reset stream position to the start
        preprocessor = joblib.load(stream)
        return preprocessor

    except Exception as e:
        print("Exception in reading preprocessor from BLOB", e)
        raise
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def read_data_from_blob(dataset_name):
    try:
        content = container_client.download_blob(dataset_name).content_as_text() #encoding='latin-1'
        data = pd.read_csv(StringIO(content))
        return data
    except Exception as e:
        print("Exception in reading dataframe from BLOB", e)
        raise HTTPException(status_code=404, detail='Error in reading dataframe from blob storage')
