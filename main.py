from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import pandas as pd
import random
from pydantic import BaseModel
import joblib


from object_klasifikasi import predict_image, read_imagefile

dataset_path = "nutrition_recom.csv"
model_path = "kmeans_model.joblib"

class InputData(BaseModel):
    berat_badan: float
    tinggi_badan: float
    pantangan: str

app = FastAPI(title='Giziku!')

@app.get("/")
async def read_root():
    return {"Giziku-API"}

@app.post("/predict", status_code=200)
async def predict_img(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg","jpeg","png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict_image(image)
    
    return {"data": prediction}

@app.post("/rekomendasi/", status_code=200)
def get_rekomendasi(data: InputData):
    try:
        dataset = pd.read_csv(dataset_path)
        dataset_cleaned = dataset.dropna()
        X_numeric = dataset_cleaned[['calories', 'proteins', 'fat', 'carbohydrate']]
        kmeans = joblib.load(model_path)
        y_kmeans = kmeans.predict(X_numeric)

        cluster_labels = {
            0: "rendah-protein-rendah-lemak",
            1: "rendah-protein-tinggi-lemak",
            2: "tinggi-protein-rendah-lemak",
            3: "tinggi-protein-tinggi-lemak",
            4: "lainnya"
        }

        cluster_data = {}
        for label in set(y_kmeans):
            cluster_items = dataset_cleaned[y_kmeans == label]["name"].tolist()
            cluster_data[cluster_labels[label]] = cluster_items

        berat_badan = data.berat_badan
        tinggi_badan = data.tinggi_badan / 100
        pantangan_list = [item.strip().lower() for item in data.pantangan.split(",")]

        bmi = berat_badan / (tinggi_badan ** 2)
        
        rekomendasi = []
        if bmi < 18.5:
            rekomendasi.extend(["rendah-protein-tinggi-lemak", "tinggi-protein-tinggi-lemak"])
        elif 18.5 <= bmi <= 24.9:
            rekomendasi.extend(["rendah-protein-rendah-lemak", "rendah-protein-tinggi-lemak", "tinggi-protein-rendah-lemak"])
        else:
            rekomendasi.append("tinggi-protein-rendah-lemak")

        filtered_rekomendasi = []
        for rekom in rekomendasi:
            filtered_items = [item for item in cluster_data[rekom] if all(pantangan not in item.lower() for pantangan in pantangan_list)]
            if filtered_items:
                filtered_rekomendasi.append({rekom: random.sample(filtered_items, min(10, len(filtered_items)))})

        if not filtered_rekomendasi:
            raise HTTPException(status_code=404, detail="Tidak ada rekomendasi makanan yang sesuai.")

        return {"BMI": bmi, "rekomendasi": filtered_rekomendasi}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, debug=True)