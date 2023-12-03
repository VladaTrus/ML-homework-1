import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

from fastapi.responses import FileResponse
from io import BytesIO
import pickle


app = FastAPI()


with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('one.pkl', 'rb') as f:
    one = pickle.load(f)

with open('cat_columns.pkl', 'rb') as f:
    cat_columns = pickle.load(f)

with open('rgd.pkl', 'rb') as f:
    rgd = pickle.load(f)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def prepare_data(df: pd.DataFrame):
    if 'selling_price' in df.columns:
        df.drop(['selling_price'], axis=1, inplace=True)
    if 'name' in df.columns:
        df.drop(['name'], axis=1, inplace=True)
    

    df_tmp = df[df['torque'].notna()]['torque']
    df_tmp = df_tmp.str.replace(',', '.')
    match_3 = df_tmp.str.extract('(\d+\.\d+|\d+)[^.\d]+(\d+\.\d+|\d+)[^.\d]+(\d+\.\d+|\d+)')
    match_2 = df_tmp.str.extract('(\d+\.\d+|\d+)[^.\d]+(\d+\.\d+|\d+)')
    match_1 = df_tmp.str.extract('(\d+\.\d+|\d+)')
    match_3[0] = match_3[0].fillna(match_2[0])
    match_3[1] = match_3[1].fillna(match_2[1])
    match_3[0] = match_3[0].fillna(match_1[0])
    match_3[1] = match_3[1].str.replace('.', '').astype(float)
    match_3[2] = match_3[2].str.replace('.', '').astype(float)
    match_3.loc[df_tmp.str.lower().str.contains('kgm'), 0] = match_3[0].astype(float) * 9.81
    match_3['torque'] = match_3[0]
    match_3['max_torque_rpm'] = match_3[[1,2]].max(axis=1)
    df.drop(['torque'], axis=1, inplace=True)
    df['max_torque_rpm'] = match_3['max_torque_rpm']
    df['torque'] = match_3['torque'].astype(float)
    
    
    df['mileage'] = df['mileage'].str.split(' ', expand=True)[0].astype(float)
    df['engine'] = df['engine'].str.split(' ', expand=True)[0].astype(float)
    df['max_power'] = df['max_power'].str.split(' ', expand=True)[0].replace('', None).astype(float)


    with open('gaps_to_median.pkl', 'rb') as f:
        gaps_med = pickle.load(f)
    
    gaps_med = pd.Series(gaps_med)
    df[gaps_med.index] = df[gaps_med.index].fillna(pd.Series(gaps_med))
    
    num_features = df.select_dtypes(include='number').columns
    df[num_features] = scaler.transform(df[num_features])
    
    cat_features = df.select_dtypes(include='object').columns
    res = one.transform(df[cat_features]).toarray()
    
    df.drop(cat_features, axis=1, inplace=True)
    df = pd.concat([df, pd.DataFrame(columns=cat_columns, data=res)], axis=1)

    df['power/engine'] = df['max_power'] / df['engine']
    df['year_squared'] = df['year'] ** 2
    
    return df

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    df = prepare_data(df)
    res = rgd.predict(df)
    res = np.round(np.exp(res) - 1)
    return res

@app.post("/predict_items")
def predict_items(file: UploadFile) -> FileResponse:
    data = file.file.read()
    df = pd.read_csv(BytesIO(data))
    file.file.close()
    df_pred = prepare_data(df.copy())
    res = rgd.predict(df_pred)
    res = np.round(np.exp(res) - 1)
    df["prediction"] = res
    df.to_csv("preds.csv", index=False)
    file_resp = FileResponse(
        path="preds.csv", media_type="text/csv", filename="preds.csv"
    )
    return file_resp