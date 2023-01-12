from typing import Dict, List
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from quran_search import SearchEngine

app = FastAPI()
templates = Jinja2Templates(directory='templates')
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

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


@app.get("/search")
def form_post(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})

@app.post("/search")
async def form_post(request: Request, query_txt: str = Form(...)):
    predictions = search_engine.run_search(query_txt)

    return templates.TemplateResponse('results.html', context={'request': request,
                                                             'query': query_txt,
                                                             'predictions': predictions})

@app.get("/")
def home():
    return({"message": "System is up"})
