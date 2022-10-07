import pandas as pd
from flask import Flask, render_template, request
from glob import glob
from tqdm import tqdm

app = Flask(__name__)

input_foder = "/data/DeepLearning/mehdi/extract/"
list_csv = glob(input_foder + "*")[:]
df_list = []
for i in tqdm(range(len(list_csv))):
    df_list.append(pd.read_csv(list_csv[i]))
df = pd.concat(df_list, ignore_index=True)


@app.route("/")
def hello_world():
    list_slide = list(df["slide"].unique())

    return render_template("index_annot.html", list_slide=list_slide)


@app.route("/ft", methods=["GET"])
def api_prediction():
    file_name = request.args.get("file_name")
    df_train = df[df["slide"] == file_name]
    df_train = df_train[df_train["image"].notna()]

    # mask = df_train["area"] < 115
    # df_train = df_train[mask]

    # df_train = df_train.sample(len(df_train))
    # images_nuc = df_train["image_nuc"]
    images = df_train["image"]
    area = df_train.area.values
    hovernet_pred = df_train.type_prob.values
    # model_name = "/data/DeepLearning/mehdi/top_gear/epoch=10-val_loss_ce=0.000.ckpt"
    return {
        "images": list(images.values),
        # "images_nuc": list(images_nuc.values),
        "index": list(df_train.index),
        "area": list(area),
        "hovernet_pred": list(hovernet_pred),
    }
