from glob import glob
import json
from flask import Flask, render_template, Response, request
import numpy as np
import pandas as pd

from predict import (
    calculation,
    export_img_cv2,
    gradcam,
    patches,
    predict,
    statis,
    top,
)
from pathaia.util.types import Slide

app = Flask(__name__)


@app.route("/")
def hello_world():
    list_slide = (
        [p.split("/")[-1] for p in glob("/media/AprioricsSlides" + "/*" + "-1_*")]
        # + [p.split("/")[-1] for p in glob("/media/AprioricsSlides" + "/*" + "-1-?-1_*")]
        + [
            "VJ_cohorte_test/" + p.split("/")[-1]
            for p in glob("/media/AprioricsSlides/VJ_cohorte_test/*")
        ]
    )
    # print(glob("/media/AprioricsSlid/es/*"))
    file_name = "/home/mehdi/code/luminal/test.json"
    model_list = [p for p in glob("/data/DeepLearning/mehdi/top_gear/*")]
    # with open(file_name, "r") as f:
    #     slide_thum = json.load(f)
    # slide_thum = ["data:image/png;base64," + img for img in slide_thum]

    return render_template("index.html", model_list=model_list, list_slide=list_slide)


@app.route("/ft")
def hi():
    list_slide = [
        p.split("/")[-1].split(".")[0]
        for p in glob("/home/mehdi/code/luminal/data/geojson_lum" + "/*")
    ]
    # file_name = "/home/mehdi/code/luminal/test.json"
    # model_list = [p for p in glob("/data/DeepLearning/mehdi/top_gear/*")]
    # with open(file_name, "r") as f:
    # slide_thum = json.load(f)
    # slide_thum = ["data:image/png;base64," + img for img in slide_thum]

    return render_template(
        "feature_index.html",
        list_slide=list_slide,
    )


@app.route("/prediction", methods=["GET"])
def api_prediction():
    file_name = request.args.get("file_name")
    model_name = request.args.get("model_name")
    print(model_name)
    # model_name = "/data/DeepLearning/mehdi/top_gear/epoch=10-val_loss_ce=0.000.ckpt"
    return predict(model_name, file_name)


@app.route("/shuffle", methods=["GET"])
def api_shuffle():
    file_name = request.args.get("file_name")
    model_name = request.args.get("model_name")
    return predict(model_name, file_name)["top"]


@app.route("/gradcam", methods=["GET"])
def api_gradcam():
    file_name = request.args.get("file_name")
    x = float(request.args.get("x"))
    y = float(request.args.get("y"))
    model_name = request.args.get("model_name")

    return gradcam(x, y, file_name, model_name)


@app.route("/patches", methods=["GET"])
def api_patches():
    file_name = request.args.get("file_name")
    file_name_ = request.args.get("file_name_")
    print(file_name, file_name_)

    return patches(file_name, file_name_)


@app.route("/shuffles", methods=["GET"])
def api_patchess():
    file_name = request.args.get("file_name")
    file_name_ = request.args.get("file_name_")
    print(file_name, file_name_)

    return patches(file_name, file_name_)


@app.route("/stat", methods=["GET"])
def api_stat():
    file_name = request.args.get("file_name")
    x = float(request.args.get("x"))
    y = float(request.args.get("y"))
    df = pd.read_csv("/home/mehdi/code/luminal/21I000004-1-03-1_135435.csv")
    # df = calculation(file_name, x, y)

    return statis(df)
