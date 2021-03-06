import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from scania.model.load_production_model import Load_Prod_Model
from scania.model.prediction_from_model import Prediction
from scania.model.training_model import Train_Model
from scania.validation_insertion.prediction_validation_insertion import Pred_Validation
from scania.validation_insertion.train_validation_insertion import Train_Validation
from utils.main_utils import upload_logs
from utils.read_params import read_params

app = FastAPI()

config = read_params()

templates = Jinja2Templates(directory=config["templates"]["dir"])

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        config["templates"]["index_html_file"], {"request": request}
    )


@app.get("/train")
async def trainRouteClient():
    try:
        train_val = Train_Validation(config["s3_bucket"]["scania_raw_data"])

        train_val.training_validation()

        train_model = Train_Model()

        num_clusters = train_model.training_model()

        load_prod_model = Load_Prod_Model(num_clusters=num_clusters)

        load_prod_model.load_production_model()

        upload_logs("logs", config["s3_bucket"]["input_files"])

        return Response("Training successfull!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.get("/predict")
async def predictRouteClient():
    try:
        pred_val = Pred_Validation(config["s3_bucket"]["scania_raw_data"])

        pred_val.prediction_validation()

        pred = Prediction()

        bucket, filename, json_predictions = pred.predict_from_model()

        return Response(
            f"prediction file created in {bucket} bucket with filename as {filename}, and few of the predictions are {str(json.loads(json_predictions))}"
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    host = config["app"]["host"]

    port = config["app"]["port"]

    uvicorn.run(app, host=host, port=port)
