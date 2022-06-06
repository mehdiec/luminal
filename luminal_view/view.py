from glob import glob
import json
from flask import Flask, render_template, Response, request
import numpy as np

from predict import export_img_cv2, gradcam, predict, top
from pathaia.util.types import Slide

app = Flask(__name__)


@app.route("/")
def hello_world():
    list_slide = [
        p.split("/")[-1] for p in glob("/media/AprioricsSlides" + "/*" + "-1-??-1_*")
    ]
    file_name = "/home/mehdi/code/luminal/test.json"
    with open(file_name, "r") as f:
        slide_thum = json.load(f)
    # slide_thum = ["data:image/png;base64," + img for img in slide_thum]

    return render_template("test.html", list_slide=list_slide, slide_thum=slide_thum)


@app.route("/prediction", methods=["GET"])
def api_prediction():
    file_name = request.args.get("file_name")
    model_name = "/data/DeepLearning/mehdi/log/luminal/resnet_319/luminal/15a61c98fef74769ac047e1ba1654c66/checkpoints/epoch=10-val_loss_ce=0.000.ckpt"
    return predict(model_name, file_name)


@app.route("/shuffle", methods=["GET"])
def api_shuffle():
    file_name = request.args.get("file_name")
    model_name = "/data/DeepLearning/mehdi/log/luminal/resnet_319/luminal/15a61c98fef74769ac047e1ba1654c66/checkpoints/epoch=10-val_loss_ce=0.000.ckpt"
    return predict(model_name, file_name)["top"]


@app.route("/gradcam", methods=["GET"])
def api_gradcam():
    file_name = request.args.get("file_name")
    x = float(request.args.get("x"))
    y = float(request.args.get("y"))
    model_name = "/data/DeepLearning/mehdi/log/luminal/resnet_319/luminal/15a61c98fef74769ac047e1ba1654c66/checkpoints/epoch=10-val_loss_ce=0.000.ckpt"
    return gradcam(x, y, file_name, model_name)
