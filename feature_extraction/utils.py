import base64
import cv2
import mahotas
import matplotlib.pyplot as plt
import numpy as np
import os
import xgboost

from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
import skimage.color


def apply_brightness_contrast(input_img, brightness=0, contrast=0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def generate_unique_logpath(logdir, raw_run_name):
    """Verify if the path already exist

    Args:
        logdir (str): path to log dir
        raw_run_name (str): name of the file

    Returns:
        str: path to the output file
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def coarseness(image, kmax=5):
    image = np.array(image)
    w = image.shape[0]
    h = image.shape[1]
    if w == 0 or h == 0:
        return np.nan
    kmax = kmax if (np.power(2, kmax) < w) else int(np.log(w) / np.log(2))
    kmax = kmax if (np.power(2, kmax) < h) else int(np.log(h) / np.log(2))
    average_gray = np.zeros([kmax, w, h])
    horizon = np.zeros([kmax, w, h])
    vertical = np.zeros([kmax, w, h])
    Sbest = np.zeros([w, h])

    for k in range(kmax):
        window = np.power(2, k)
        for wi in range(w)[window : (w - window)]:
            for hi in range(h)[window : (h - window)]:
                average_gray[k][wi][hi] = np.sum(
                    image[wi - window : wi + window, hi - window : hi + window]
                )
        for wi in range(w)[window : (w - window - 1)]:
            for hi in range(h)[window : (h - window - 1)]:
                horizon[k][wi][hi] = (
                    average_gray[k][wi + window][hi] - average_gray[k][wi - window][hi]
                )
                vertical[k][wi][hi] = (
                    average_gray[k][wi][hi + window] - average_gray[k][wi][hi - window]
                )
        horizon[k] = horizon[k] * (1.0 / np.power(2, 2 * (k + 1)))
        vertical[k] = horizon[k] * (1.0 / np.power(2, 2 * (k + 1)))

    for wi in range(w):
        for hi in range(h):
            h_max = np.max(horizon[:, wi, hi])
            h_max_index = np.argmax(horizon[:, wi, hi])
            v_max = np.max(vertical[:, wi, hi])
            v_max_index = np.argmax(vertical[:, wi, hi])
            index = h_max_index if (h_max > v_max) else v_max_index
            Sbest[wi][hi] = np.power(2, index)

    fcrs = np.mean(Sbest)
    return fcrs


def contrast(image):
    image = np.array(image)
    image = np.reshape(image, (1, image.shape[0] * image.shape[1]))
    m4 = np.mean(np.power(image - np.mean(image), 4))
    v = np.var(image)
    std = np.power(v, 0.5)
    alfa4 = m4 / np.power(v, 2)
    fcon = std / np.power(alfa4, 0.25)
    return fcon


def directionality(image):
    image = np.array(image, dtype="int64")
    h = image.shape[0]
    w = image.shape[1]
    if w == 0 or h == 0:
        return np.nan
    convH = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    convV = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    deltaH = np.zeros([h, w])
    deltaV = np.zeros([h, w])
    theta = np.zeros([h, w])

    # calc for deltaH
    for hi in range(h)[1 : h - 1]:
        for wi in range(w)[1 : w - 1]:
            deltaH[hi][wi] = np.sum(
                np.multiply(image[hi - 1 : hi + 2, wi - 1 : wi + 2], convH)
            )
    for wi in range(w)[1 : w - 1]:
        deltaH[0][wi] = image[0][wi + 1] - image[0][wi]
        deltaH[h - 1][wi] = image[h - 1][wi + 1] - image[h - 1][wi]
    for hi in range(h):
        deltaH[hi][0] = image[hi][1] - image[hi][0]
        deltaH[hi][w - 1] = image[hi][w - 1] - image[hi][w - 2]

    # calc for deltaV
    for hi in range(h)[1 : h - 1]:
        for wi in range(w)[1 : w - 1]:
            deltaV[hi][wi] = np.sum(
                np.multiply(image[hi - 1 : hi + 2, wi - 1 : wi + 2], convV)
            )
    for wi in range(w):
        deltaV[0][wi] = image[1][wi] - image[0][wi]
        deltaV[h - 1][wi] = image[h - 1][wi] - image[h - 2][wi]
    for hi in range(h)[1 : h - 1]:
        deltaV[hi][0] = image[hi + 1][0] - image[hi][0]
        deltaV[hi][w - 1] = image[hi + 1][w - 1] - image[hi][w - 1]

    deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
    deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

    # calc the theta
    for hi in range(h):
        for wi in range(w):
            if deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0:
                theta[hi][wi] = 0
            elif deltaH[hi][wi] == 0:
                theta[hi][wi] = np.pi
            else:
                theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
    theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

    n = 16
    t = 12
    cnt = 0
    hd = np.zeros(n)
    dlen = deltaG_vec.shape[0]
    for ni in range(n):
        for k in range(dlen):
            if (
                (deltaG_vec[k] >= t)
                and (theta_vec[k] >= (2 * ni - 1) * np.pi / (2 * n))
                and (theta_vec[k] < (2 * ni + 1) * np.pi / (2 * n))
            ):
                hd[ni] += 1
    hd = hd / np.mean(hd)
    hd_max_index = np.argmax(hd)
    fdir = 0
    for ni in range(n):
        fdir += np.power((ni - hd_max_index), 2) * hd[ni]
    return fdir


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print(
            "\n Time taken: %i hours %i minutes and %s seconds."
            % (thour, tmin, round(tsec, 2))
        )


def preprocess(df, y_predicition):
    indices = [int(elt) for elt in list((y_predicition).keys())]
    df["pred_deep"] = np.nan
    df["pred_deep"].iloc[indices] = list(y_predicition.values())

    if df["ee_mitoses_par_mm2"].dtype == "O":
        df["ee_mitoses_par_mm2"] = (
            df["ee_mitoses_par_mm2"].str.replace(",", ".").astype(float)
        )

    df["rp_intensite"][df["rp_intensite"] == "0"] = "+"
    df["rp_intensite"][df["rp_intensite"] == "+/++/+++"] = "++"
    df["nb_ggl_rc"][df["nb_ggl_rc"] == "na"] = "0"
    df["nb_ggl_rc"] = df["nb_ggl_rc"].astype(int)

    df["rp_intensite"][df["rp_intensite"] == 0] = "+"
    df["rp_intensite"][df["rp_intensite"] == "+/+++"] = "++"
    df["pN"][df["pN"] == "pN1(mi)"] = "pN1mi"
    df["pN"][df["pN"] == "pN"] = "pN0"
    # creating initial dataframe

    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    # Assigning numerical values and storing in another column
    labelencoder.fit(["+", "+/++", "++", "++/+++", "+++"])
    df["re_intensite_cat"] = labelencoder.transform(df["re_intensite"])
    df["rp_intensite_cat"] = labelencoder.transform(df["rp_intensite"])

    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    # Assigning numerical values and storing in another column
    labelencoder.fit(["pN0", "pN0(i+)", "pN1mi", "pN1a", "pN2"])
    df["pN_cat"] = labelencoder.transform(df["pN"])

    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    # Assigning numerical values and storing in another column
    labelencoder.fit(
        [
            "luminal A",
            "luminal B",
        ]
    )
    df["ab_cat"] = labelencoder.transform(df["ab"])

    return df


def show_results(df, df_test, columns, skf, clf, name="", params={}):
    full_acc = []
    full_auc = []

    y = df["ab_cat"]
    x = df[columns]
    x = (x - x.min()) / (x.max() - x.min())

    for train_index, test_index in skf.split(x, y):

        x_train, x_test, y_train, y_test = (
            x.iloc[train_index],
            x.iloc[test_index],
            y.iloc[train_index],
            y.iloc[test_index],
        )

        ac, auc = train_pred(clf, x_train, x_test, y_train, y_test)
        full_acc.append(ac)
        full_auc.append(auc)

    full_acc = np.array(full_acc)
    full_auc = np.array(full_auc)
    print(full_acc.mean(), full_auc.mean())

    y_train = df["ab_cat"]
    x_train = df[columns]

    y_test = df_test["ab_cat"]
    x_test = df_test[columns]
    x_test = (x_test - x_train.min()) / (x_train.max() - x_train.min())
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    if params:

        clf = xgboost.XGBClassifier(
            **params.best_params_,
            random_state=202,
        )
    else:
        clf = xgboost.XGBClassifier(
            random_state=202,
        )

    clf.fit(x_train, y_train)

    with open(f"{name}.npy", "wb") as f:
        np.save(f, clf.predict_proba(x_test)[:, 1])

    train_pred(clf, x_train, x_test, y_train, y_test)

    plot_roc(x_test, y_test, clf)


def plot_roc(x_test, y_test, clf):

    lw = 2
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(x_test)[:, 1])

    aucs = roc_auc_score(y_test, clf.predict(x_test))
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % aucs,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # plt.savefig(f"./zfigure/ROC_slide_test_fold_{i}.png")
    # plt.savefig(f"./zfigure/ROC_slide_test_fold_{i}.pdf")
    plt.show()


def train_pred(clf, x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train)
    preds = clf.predict(x_train)
    predss = clf.predict(x_test)

    print(len(x_train), len(x_test))

    # random forest classifier with n_estimators=10 (default)

    # print(clf.predict(x_test))
    acc = accuracy_score(y_test, predss)
    print("Accuracy test is: ", acc)
    ac = accuracy_score(y_train, preds)
    print("Accuracy train is: ", ac)
    ac = precision_score(y_test, predss)
    print("precision test is: ", ac)
    ac = precision_score(y_train, preds)
    print("precision train is: ", ac)
    # print(y_test)

    auc = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
    print("auc teste is: ", auc)

    return acc, auc


def red(nuc):
    return np.mean(nuc[:, :, 0].flatten())


def green(nuc):
    return np.mean(nuc[:, :, 0].flatten())


def blue(nuc):
    return np.mean(nuc[:, :, 0].flatten())


def h_mean(nuc):
    he_image = skimage.color.rgb2hed(np.array(nuc))
    return np.mean(he_image[:, :, 0].flatten())


def e_mean(nuc):
    he_image = skimage.color.rgb2hed(np.array(nuc))
    return np.mean(he_image[:, :, 1].flatten())


def h_std(nuc):
    he_image = skimage.color.rgb2hed(np.array(nuc))
    return np.std(he_image[:, :, 0].flatten())


def e_std(nuc):
    he_image = skimage.color.rgb2hed(np.array(nuc))
    return np.std(he_image[:, :, 1].flatten())


def hue_mean(nuc):
    lab_image = skimage.color.rgb2hsv(np.array(nuc))
    return np.mean(lab_image[:, :, 0].flatten())


def saturation_mean(nuc):
    lab_image = skimage.color.rgb2hsv(np.array(nuc))
    return np.mean(lab_image[:, :, 1].flatten())


def value_hsv_mean(nuc):
    lab_image = skimage.color.rgb2hsv(np.array(nuc))
    return np.mean(lab_image[:, :, 2].flatten())


def hue_std(nuc):
    lab_image = skimage.color.rgb2hsv(np.array(nuc))
    return np.std(lab_image[:, :, 0].flatten())


def saturation_std(nuc):
    lab_image = skimage.color.rgb2hsv(np.array(nuc))
    return np.std(lab_image[:, :, 1].flatten())


def value_hsv_std(nuc):
    lab_image = skimage.color.rgb2hsv(np.array(nuc))
    return np.std(lab_image[:, :, 2].flatten())


def in_patch(centroids, countour_patch):
    return (
        cv2.pointPolygonTest(
            countour_patch,
            (centroids[0], centroids[1]),
            False,
        )
        == 1
    )


def get_hu(gray):
    return cv2.HuMoments(cv2.moments(gray)).flatten()


def glcm(gscale_img):
    if gscale_img.shape[0] > 2 and gscale_img.shape[1] > 2:
        return mahotas.features.haralick(
            gscale_img.astype(int), ignore_zeros=True
        ).mean(axis=0)
    else:
        return


def bounding_box(bboxs, x_patch, y_patch):
    xx, yy = np.array(bboxs[0]) - np.array([x_patch, y_patch])
    w, h = np.array(bboxs[1]) - np.array([x_patch, y_patch])
    return xx, yy, w, h


def valid_bbox(contour, p_size):
    xx, yy, w, h = contour
    return ~(
        w > p_size
        or h > p_size
        or xx > p_size
        or yy > p_size
        or w < 0
        or h < 0
        or xx < 0
        or yy < 0
    )


def elipsea(contour_centered):

    try:

        _, l, _ = cv2.fitEllipse(contour_centered)
        return max(l)
    except:
        return 1


def elipseb(contour_centered):

    try:

        _, l, _ = cv2.fitEllipse(contour_centered)
        return min(l)
    except:
        return 1


def radiuses(contour):
    xx, yy, w, h = contour
    return max(w - xx, h - yy) / 2


def convex_area(contour):
    return cv2.contourArea(cv2.convexHull(contour).squeeze())


def contour_centered(contour, x_patch, y_patch):
    return np.array(contour, dtype=np.int32) - np.array([x_patch, y_patch])


def get_gray_scale_nuc(bbox_centered, image, p_size):

    x, y, w, h = np.abs(bbox_centered)

    gray_0 = np.zeros((h, w))
    # gray_0 = np.array([])
    # gray_0 = IMAGE_GRAY.copy()
    # show =  gray_0[ y:y+h,  x:   x+w ]
    if h == 0 or w == 0 or x + w > p_size or y + h > p_size:
        return np.array([[1, 1], [1, 1]])
    np.copyto(gray_0, image[y : y + h, x : x + w])

    # return mahotas.features.haralick(
    #                 show.astype(int)
    #             ).mean(axis=0)
    return gray_0


def get_color_nuc(bbox_centered, image, p_size):

    x, y, w, h = np.abs(bbox_centered)

    gray_0 = np.zeros((h, w, 3))
    # gray_0 = np.array([])
    # gray_0 = IMAGE_GRAY.copy()
    # show =  gray_0[ y:y+h,  x:   x+w ]
    if h == 0 or w == 0 or x + w > p_size or y + h > p_size:
        return np.zeros((4, 4, 3))
    np.copyto(gray_0, image[y : y + h, x : x + w])

    # return mahotas.features.haralick(
    #                 show.astype(int)
    #             ).mean(axis=0)
    return gray_0


def length(c):
    return cv2.arcLength(c, True)


def get_displayde_imaged(col, image, p_size, dezoom=30):
    x, y, w, h = np.abs(col.bbox_centered)
    # image_tmp = np.zeros((h, w, 3))
    image_tmp = np.zeros((p_size, p_size, 3))
    dezoom_y = max(h, dezoom)
    dezoom_x = max(w, dezoom)
    full = p_size - 1
    ymin = max(y - dezoom_y, 0)
    ymax = min(y + dezoom_y, full)
    xmin = max(x - dezoom_x, 0)
    xmax = min(x + dezoom_x, full)

    np.copyto(image_tmp, image)
    _ = cv2.drawContours(image_tmp, np.array([col.contour_centered]), 0, (0, 0, 0), 0)

    if ymin == 0:
        ymax = 2 * dezoom_y
    if ymax == full:
        ymin = full - 2 * dezoom_y
    if xmin == 0:
        xmax = 2 * dezoom_x
    if xmax == full:
        xmin = full - 2 * dezoom_x

    return image_tmp[ymin:ymax, xmin:xmax]


def export_img_cv2(img):
    _, buff = cv2.imencode(".png", img)
    return base64.b64encode(buff).decode("ascii")
