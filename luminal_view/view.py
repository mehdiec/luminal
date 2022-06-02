from glob import glob
from flask import Flask, render_template, Response, request

from predict import gradcam, predict

app = Flask(__name__)


@app.route("/")
def hello_world():
    list_slide = [
        p.split("/")[-1] for p in glob("/media/AprioricsSlides" + "/*" + "-1-??-1_*")
    ]
    return render_template("index.html", list_slide=list_slide)


@app.route("/prediction", methods=["GET"])
def api_prediction():
    file_name = request.args.get("file_name")
    model_name = "/data/DeepLearning/mehdi/log/luminal/resnet_319/luminal/15a61c98fef74769ac047e1ba1654c66/checkpoints/epoch=20-val_loss_ce=0.000.ckpt"
    return predict(model_name, file_name)


@app.route("/gradcam", methods=["GET"])
def api_gradcam():
    file_name = request.args.get("file_name")
    x = float(request.args.get("x"))
    y = float(request.args.get("y"))
    model_name = "/data/DeepLearning/mehdi/log/luminal/resnet_319/luminal/15a61c98fef74769ac047e1ba1654c66/checkpoints/epoch=20-val_loss_ce=0.000.ckpt"
    return gradcam(x, y, file_name, model_name)
