from typing import Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from quran_search import SearchEngine

app = FastAPI()
search_engine = SearchEngine()

class QueryText(BaseModel):
    text: str

class PredictionObject(BaseModel):
    ayah_txt: str
    ayah_num: int
    surah_name: str

class PredictionsObject(BaseModel):
    predictions: List[PredictionObject]


@app.post("/predict", summary="search through holy quran")
async def predict(query_text: QueryText):
    try:
        predictions = search_engine.run_search(query_text.text)
        predictions_objects = PredictionsObject(predictions = predictions)
        return predictions_objects
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/")
def home():
    return({"message": "System is up"})
