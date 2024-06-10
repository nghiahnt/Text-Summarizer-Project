from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

class DataRequest(BaseModel):
    text: str
    levelValue: int

text:str = "What is Text Summarization?"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    



@app.post("/predict")
async def predict_route(text):
    try:

        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e
    

@app.post("/process")
async def process_route(data: DataRequest):
    try:
        obj = PredictionPipeline()
        result = obj.predict(data.text, data.levelValue)
        return result

    except Exception as e:
        raise e


if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8080)
