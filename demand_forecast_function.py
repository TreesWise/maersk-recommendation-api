import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
from statsmodels.tsa.seasonal import STL
import pickle
from pmdarima.arima import auto_arima
import tensorflow as tf
from joblib import load
from azure.storage.blob import ContainerClient
from io import StringIO
import json
import numpy as np
from io import BytesIO

import warnings
warnings.filterwarnings('ignore')

container_client = ContainerClient.from_connection_string(
    'DefaultEndpointsProtocol=https;AccountName=treewiseblobstorage;AccountKey=jE3f/ogf+EH2cZyJEEagULdbWrIFvtKOnJB655pvrSn+9jzniIx8hGjHlBvnb3Py2I6h7b5zO2NO+AStfk0NPA==;EndpointSuffix=core.windows.net', container_name='maersk-vendor-recommendation-db')
def read_data_from_blob(dataset_name):
    try:
        content = container_client.download_blob(
            dataset_name).content_as_text(encoding='latin-1')
        data = pd.read_csv(StringIO(content))
        return data
    except Exception as e:
        print("Exception in reading from BLOB", e)
        raise HTTPException(
            status_code=404, detail='Error in reading data from blob storage')
data = read_data_from_blob("IMPA_ITEMS_WITH_PORT.csv")
# print(data)        
def Demand_forecast(df, vessel_type, vessel_sub_type):
    print("vessel_sub_type", vessel_sub_type)
    with open('final_res2.pickle', 'rb') as file1:
        util = pickle.load(file1)
    print('pic')
    # Data reading and processing
    # path = r'D:\envs\demand_forecasting\Codes\2.Demand_Forecasting_Items_Victualling_11_01_2023/'
    # df = pd.read_csv('2.Demand_Forecasting_Items_Victualling_11_01_2023.csv')
    # removing type-in section
    df = df[(df['ITEM_SECTION_1'] == 'PROVISION') | (
        df['ITEM_SECTION_1'] == 'SEA CHEF PROVISIONS')]
    section_from_item = ['PROVISION TOTAL', 'PROVISION', 'BONDED STORES']
    df = df[~df['ITEM'].isin(section_from_item)]
    # date formating
    df['APPROVED_DATE'] = pd.to_datetime(df['APPROVED_DATE'])
    df.sort_values(by=['APPROVED_DATE'], ascending=True, inplace=True)
    df = df[(df['APPROVED_DATE'] > '2020-12-31') &
            (df['APPROVED_DATE'] <= '2023-12-31')]
    df['Year_Quarter'] = df['APPROVED_DATE'].dt.to_period('Q')
    #
    df.dropna(subset=['UNIT_PRICE', 'PO_QTY'], how='any', inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df_qtr = pd.DataFrame(df.groupby(['Year_Quarter', 'VESSEL_TYPE', 'VESSEL_SUB_TYPE', 'ITEM'])['PO_QTY'].sum()).reset_index()  # ,'VESSEL_TYPE','VESSEL_SUB_TYPE',
    print(df_qtr.columns)
    df_qtr = df_qtr[(df_qtr['VESSEL_TYPE'] == vessel_type)&(df_qtr['VESSEL_SUB_TYPE'] == vessel_sub_type)]
    date_rng = pd.date_range(start='2021-01-01', end='2023-12-31', freq='M')
    check_qtr = pd.DataFrame({'Date': date_rng})
    # Prediction_results
    qty_pred = {}
    acc_metrics = util['acc_metrics']
    items_req = dict(sorted({k: v for k, v in acc_metrics.items() if v <= 50}.items(
    ), key=lambda x: x[1], reverse=True)).keys()  # 246 >100
    # items_req = #need input
    print('arima started')
    for its in items_req:
        item_df = df_qtr[df_qtr['ITEM'] == its]
        for q1 in list(check_qtr['Date'].dt.to_period('Q').unique()):
            if q1 in item_df['Year_Quarter'].values:
                pass
            else:
                item_df = pd.concat([item_df, pd.DataFrame(
                    {'Year_Quarter': q1, 'ITEM': its, 'PO_QTY': 1}, index=[0])], axis=0)
        item_df.reset_index(drop=True, inplace=True)
        item_df.set_index(item_df['Year_Quarter'], drop=True, inplace=True)
        item_df.drop(columns=['Year_Quarter'], inplace=True)
        series = pd.Series(item_df['PO_QTY'], name=its)
        series.sort_index(inplace=True, ascending=True)
        series = series.astype(int)
        series.replace({0: 1}, inplace=True)
        if len(series.unique()) == 1:
            # less_items.append(its)
            pass
        else:
            # lambdax = float(util['box_cox'][its].split('_')[0])#need input
            period = util['box_cox'][its].split('_')[1]  # need input
            fitted_data, lambdax = stats.boxcox(series.values)
            stl1 = STL(fitted_data, period=int(period), robust=True)
            results = stl1.fit()
            # predicting seasonality
            aa = auto_arima(results.seasonal.tolist())
            # predicting residual
            dd = auto_arima(results.resid.tolist())
            # predicting trend
            tt = auto_arima(results.trend.tolist())
            try:
                qty_pred[its] = int(abs(list(inv_boxcox(aa.predict(
                    1)+dd.predict(1)+tt.predict(1)+np.mean(results.resid), lambdax))[0]))
            except:
                pass
    # Item quantity modelling using LSTM
    lstm_features = list(dict(sorted({k: v for k, v in acc_metrics.items() if (v > 50) & (
        # need input
        v <= 100)}.items(), key=lambda x: x[1], reverse=True)).keys())
    lstm_df = pd.DataFrame()
    for l1 in lstm_features:  # df_qtr['ITEM'].unique():#
        date_rng = pd.date_range(
            start='2021-01-01', end='2023-12-31', freq='M')  # need change
        check_qtr = pd.DataFrame({'Date': date_rng})
        item_df = df_qtr[df_qtr['ITEM'] == l1]
        for q1 in list(check_qtr['Date'].dt.to_period('Q').unique()):
            if q1 in item_df['Year_Quarter'].values:
                pass
            else:
                item_df = pd.concat([item_df, pd.DataFrame(
                    {'Year_Quarter': q1, 'ITEM': l1, 'PO_QTY': 0}, index=[0])], axis=0)
        item_df.reset_index(drop=True, inplace=True)
        item_df.set_index(item_df['Year_Quarter'], drop=True, inplace=True)
        item_df.drop(columns=['Year_Quarter'], inplace=True)
        series = pd.Series(item_df['PO_QTY'], name=l1)
        series.sort_index(inplace=True, ascending=True)
        series = series.astype(int)
        lstm_df = pd.concat([lstm_df, series], axis=1)
    lstm_df.reset_index(inplace=True)
    lstm_df['quater'] = lstm_df['index'].apply(
        lambda x: int(str(x).split('Q')[1]))
    lstm_df.set_index(lstm_df['index'], inplace=True, drop=True)
    lstm_df.drop(columns=['index'], axis=1, inplace=True)

    def split_sequences(sequence_inp, sequence_out, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequence_inp)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence_inp):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence_inp[i:end_ix], sequence_out[end_ix:out_end_ix]
            pos = np.array(range(1, n_steps_in+1))/n_steps_in
            seq_x = np.append(seq_x, pos.reshape(-1, 1), axis=1)
            # nystroem_mod = Nystroem(kernel='polynomial',degree=2,random_state=1,n_components=100)
            # seq_x = nystroem_mod.fit_transform(seq_x)
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    scaler1 = load('LSTM_scaler_x.joblib')  # need input
    scaler2 = load('LSTM_scaler_y.joblib')  # need input
    scaler_x2 = scaler1.transform(lstm_df[-3:][lstm_features+['quater']])
    pos = np.array(range(1, 3+1))/3
    scaler_x2 = np.append(scaler_x2, pos.reshape(-1, 1), axis=1)
    scaler_x2 = scaler_x2.reshape(1, 3, 437)
    model = tf.keras.models.load_model('LSTM_model.h5')
    pred = pd.DataFrame(scaler2.inverse_transform(
        model.predict(scaler_x2)), columns=scaler2.feature_names_in_)
    pred = pred[util['LSTM_items']]
    qty_pred.update({m: int(abs(v[0])) for m, v in pred.to_dict().items()})
    main_dict = {'SEA CHEF PROVISIONS': {}, 'PROVISION': {}}
    sea_chef = df[df['ITEM_SECTION_1'] == 'SEA CHEF PROVISIONS']
    provision = df[df['ITEM_SECTION_1'] == 'PROVISION']
    for kk in qty_pred.keys():
        if kk in sea_chef['ITEM'].values:
            main_dict['SEA CHEF PROVISIONS'].update({kk: qty_pred[kk]})
        if kk in provision['ITEM'].values:
            main_dict['PROVISION'].update({kk: qty_pred[kk]})
    main_dict['VESSEL_TYPE'] = vessel_type
    main_dict['VESSEL_SUB_TYPE'] = vessel_sub_type
    return main_dict
Demand_forecast(data, 'Bulk Carrier', 'Bulk Carrier')