import xgboost

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC


MODEL_DICT = {
    "xgb": {
        "model": xgboost.XGBClassifier(random_state=202),
        "params": {
            "min_child_weight": [i for i in range(1, 4)],
            "gamma": [i * 0.5 for i in range(1, 4)],
            # "subsample": [i * 0.2 for i in range(4, 6)],
            # "colsample_bytree": [i * 0.2 for i in range(3, 6)],
            "max_depth": [i for i in range(1, 5)],
            "n_estimators": [
                1,
                2,
                5,
                8,
                10,
                20,
                30,
                40,
                50,
            ],
            "learning_rate": [i * 0.02 for i in range(7, 11)],
            "reg_lambda": [i * 0.1 + 1 for i in range(3, 7)],
            # "eval_metric": "auc",
        },
    },
    "rf": {
        "model": RandomForestClassifier(random_state=202),
        "params": {
            "n_estimators": [10, 20, 30, 40, 50, 60, 80, 100],
        },
    },
    "lr": {
        "model": LogisticRegression(random_state=202),
        "params": {
            "C": [i * 0.1 for i in range(1, 30)],
        },
    },
    "nusvc": {
        "model": NuSVC(random_state=202, probability=True),
        "params": {
            "nu": [i * 0.1 for i in range(1, 11)],
        },
    },
    "knn": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [i for i in range(2, 11)],
        },
    },
    "qda": {
        "model": QuadraticDiscriminantAnalysis(),
        "params": {
            "tol": [1.0e-4],
        },
    },
    "GaussianNB": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": [1e-9],
        },
    },
}
