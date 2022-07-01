from jinja2 import Template, Environment, FileSystemLoader
import pandas as pd
from loader import preprocess
import pandas as pd

from tqdm import tqdm


cfg = {
    "DATA_DIR": "/data/DeepLearning/mehdi/features",
    "DATASET": {
        "VALID_RATIO": 0,
        "PREPROCESSING": {
            "NORMALIZE": {"ACTIVE": False, "TYPE": "StandardScaler"},
            "PCA": {"ACTIVE": False, "N_COMPONENTS": 0.95},
        },
    },
}
a = preprocess(cfg)
df = pd.json_normalize(
    a["train"], "features", ["target", "transformation", "slide", "info"]
)
df = pd.json_normalize(
    a["train"], "features", ["target", "transformation", "slide", "info"]
)
df_train = df[df["transformation"] == ""]
selected_col = [
    "gd_ax",
    "pt_ax",
    "compactness",
    "length_width_ratio",
    "Eccentricity",
    "Assymetry",
    "smoothmess",
    "coarseness",
    "contrast",
    "directionality",
    "nb_bump",
    "iou",
    "h_mean",
    "h_std",
    "e_mean",
    "e_std",
    "hue_mean",
    "hue_std",
    "Angular Second Moment",
    "Contrast",
    "Correlation",
    "Sum of Squares: Variance",
    "Inverse Difference Moment",
    "Sum Average",
    "Sum Variance",
    "Sum Entropy",
    "Entropy",
    "Difference Variance",
    "Difference Entropy",
    "Information Measure of Correlation 1",
    "Information Measure of Correlation 2",
    "hu_0",
    "hu_1",
]
df_visible_area = df[df["area"] > 672]


images = {ft: [] for ft in selected_col}
for col in tqdm(selected_col):
    print(col)
    for i, value in enumerate(
        df_visible_area[col].quantile([0, 0.25, 0.5, 0.75, 1]).values
    ):

        if i == 0:
            sample = df_visible_area[df_visible_area[col] == value]["image"].values[0]
        if i == 4:
            sample = df_visible_area[df_visible_area[col] == value]["image"].values[0]
        if i > 0 and i < 4:
            sample = (
                df_visible_area[
                    (df_visible_area[col] < value) & (df_visible_area[col] > past_value)
                ]["image"]
                .sample()
                .values[0]
            )
        images[col].append("data:image/png;base64," + sample)

        past_value = value

# load templates folder to environment (security measure)
env = Environment(loader=FileSystemLoader("templates"))

# load the `index.jinja` template
index_template = env.get_template("test.jinja")
output_from_parsed_template = index_template.render(ft_atlas=images)

# write the parsed template
with open("index.html", "w") as chap_page:
    chap_page.write(output_from_parsed_template)
