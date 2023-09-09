from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from fairlearn.reductions import (
    ExponentiatedGradient,
    BoundedGroupLoss,
    ZeroOneLoss,
    GridSearch,
)
import replication_package.getdataset as getdataset
from replication_package.utils import *
from demv import DEMV
import warnings
import os
import replication_package.confusionmatrix as confusionmatrix
from os.path import exists

warnings.filterwarnings("ignore")
import argparse


parser = argparse.ArgumentParser(
    description="Metrics generator for DEMV testing.",
    epilog="Example usage: python generatemetrics.py cmc biased 3 --classifier svc",
)

parser.add_argument(
    "dataset",
    type=str,
    help="Required argument: Chosen dataset to generate metrics for. Availability of datasets changes according to the chosen method."
    + " All available datasets are: adult, cmc, compas, crime, drugs, german, obesity, park, wine.",
    choices=[
        "adult",
        "cmc",
        "law",
        "compas",
        "crime",
        "drug",
        "german",
        "obesity",
        "park",
        "wine",
        "all",
    ],
)

parser.add_argument(
    "method",
    type=str,
    help="Required argument: Chosen method to generate metrics for. Can be biased, eg, grid, uniform, smote, adasyn.",
    choices=["biased", "eg", "grid", "uniform", "smote", "adasyn"],
)

parser.add_argument(
    "number_of_features",
    type=int,
    help='Required argument: Number of sensitive features in the dataset to consider, up to 3. If "1" is chosen, two datasets will be generated, one for each canonical sensitive feature  (as described in literature for that dataset)',
    choices=[1, 2, 3, 4],
)

parser.add_argument(
    "--classifier",
    type=str,
    nargs="?",
    default="logistic",
    help="Optional argument: classifier to use. Possible options are logistic, gradient, svc and mlp. Defaults to Logistic Regression (logistic).",
    choices=["logistic", "gradient", "svc", "mlp"],
)
parser.add_argument(
    "--cm",
    action=argparse.BooleanOptionalAction,
    help="Optional argument: only generate Confusion Matrices for the selected dataset.",
)
parser.add_argument(
    "--sensitivefeature",
    type=str,
    help="Optional argument: force sensitive feature to be considered (Only works if number of features is 1)",
)

args = parser.parse_args()


def run_metrics(
    method, data, unpriv_group, sensitive_features, label, positive_label, k
):
    if method == "biased":
        model, metr = cross_val(
            pipeline,
            data,
            label,
            unpriv_group,
            sensitive_features,
            positive_label=positive_label,
        )
    elif method == "eg":
        constr = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
        exp = ExponentiatedGradient(
            pipeline, constr, sample_weight_name="classifier__sample_weight"
        )
        model, metr = cross_val(
            exp,
            data.copy(),
            label,
            unpriv_group,
            sensitive_features,
            exp=True,
            positive_label=positive_label,
        )
    elif method == "grid":
        constr = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
        exp = ExponentiatedGradient(
            pipeline, constr, sample_weight_name="classifier__sample_weight"
        )
        grid = GridSearch(
            pipeline, constr, sample_weight_name="classifier__sample_weight"
        )
        model, metr = cross_val(
            grid,
            data,
            label,
            unpriv_group,
            sensitive_features,
            positive_label,
            exp=True,
        )
    else:
        demv = DEMV(round_level=1, strategy=method)
        model, metr = cross_val(
            pipeline,
            data.copy(),
            label,
            unpriv_group,
            sensitive_features,
            debiaser=demv,
            positive_label=positive_label,
        )

    m = prepareplots(metr, "discard")

    return m


file_exists = exists("ris")
if not file_exists:
    os.makedirs("ris")
    os.makedirs("ris/1features")
    os.makedirs("ris/2features")
    os.makedirs("ris/3features")
else:
    if not exists("ris/1features"):
        os.makedirs("ris/1features")
    if not exists("ris/2features"):
        os.makedirs("ris/2features")
    if not exists("ris/3features"):
        os.makedirs("ris/3features")

file_exists = exists("confusionmatrices")
if not file_exists:
    os.makedirs("confusionmatrices")


cm = args.cm
dataset = [args.dataset]
method = [args.method]
number_of_features = min(3, args.number_of_features)
if args.classifier is not None:
    classi = args.classifier
    if classi == "logistic":
        classifier = LogisticRegression()
    elif classi == "gradient":
        classifier = GradientBoostingClassifier()
    elif classi == "svc":
        classifier = SVC()
    elif classi == "mlp":
        classifier = MLPClassifier(activation="relu", solver="adam")
else:
    classi = "logistic"
    classifier = LogisticRegression()

# Get array of sensitive features, divided by comma.
if args.sensitivefeature:
    sensitivefeature = args.sensitivefeature.split(",")
    if len(sensitivefeature) != number_of_features:
        raise Exception(
            "Mismatch between selected number of features and number of sensitive features specified."
        )
else:
    sensitivefeature = args.sensitivefeature

pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


if cm:
    dataset = dataset[0]
    confusionmatrix.generatecm(
        dataset, normalize=True, sensitivefeature=sensitivefeature
    )
    confusionmatrix.generatecm(
        dataset, debiaser="demv", normalize=True, sensitivefeature=sensitivefeature
    )
    confusionmatrix.generatecm(
        dataset, debiaser="eg", normalize=True, sensitivefeature=sensitivefeature
    )
    confusionmatrix.generatecm(
        dataset, debiaser="grid", normalize=True, sensitivefeature=sensitivefeature
    )

    sys.exit(0)

if dataset == ["all"]:
    if number_of_features != 3:
        dataset = ["cmc", "crime", "drug", "law", "obesity", "park", "wine"]
    else:
        dataset = ["cmc", "crime", "drug", "law", "wine"]

if method == ["all"]:
    method = ["biased", "demv", "smote", "adasyn", "eg", "grid"]


for m in method:
    for d in dataset:
        print("Processing " + d + " with " + m)

        # If sensitive features are specified, use them.
        if sensitivefeature:
            (
                data,
                label,
                positive_label,
                sensitive_features,
                unpriv_group,
                k,
            ) = getdataset.getdataset(
                d, number_of_features, sensitivefeature=sensitivefeature
            )
            result = run_metrics(
                m, data, unpriv_group, sensitive_features, label, positive_label, k
            )
            sf = list(sensitive_features)
            names = ""
            # Cut names of variables so file names aren't too long.
            for name in sf:
                names = names + str(name[0:6]) + "_"

            result.to_csv(
                "ris/"
                + str(number_of_features)
                + "features/metrics_"
                + d
                + "_"
                + names
                + m
                + "_"
                + str(classi)
                + ".csv"
            )

        # If sensitive features are not specified and the number_of_features is 1, generate metrics for each.
        elif number_of_features == 1:
            (
                data,
                label,
                positive_label,
                sensitive_features,
                unpriv_group,
                k,
            ) = getdataset.getdataset(d, number_of_features, singlefeature=1)
            result = run_metrics(
                m, data, unpriv_group, sensitive_features, label, positive_label, k
            )

            result.to_csv(
                "ris/"
                + str(number_of_features)
                + "features/metrics_"
                + d
                + "_"
                + str(list(sensitive_features)[0])
                + "_"
                + m
                + "_"
                + str(number_of_features)
                + "_features_"
                + str(classi)
                + ".csv"
            )

            (
                data,
                label,
                positive_label,
                sensitive_features,
                unpriv_group,
                k,
            ) = getdataset.getdataset(d, number_of_features, singlefeature=2)
            result = run_metrics(
                m, data, unpriv_group, sensitive_features, label, positive_label, k
            )

            result.to_csv(
                "ris/"
                + str(number_of_features)
                + "features/metrics_"
                + d
                + "_"
                + str(list(sensitive_features)[0])
                + "_"
                + m
                + "_"
                + str(number_of_features)
                + "_features_"
                + str(classi)
                + ".csv"
            )

        else:
            (
                data,
                label,
                positive_label,
                sensitive_features,
                unpriv_group,
                k,
            ) = getdataset.getdataset(d, number_of_features)

            result = run_metrics(
                m, data, unpriv_group, sensitive_features, label, positive_label, k
            )

            result.to_csv(
                "ris/"
                + str(number_of_features)
                + "features/metrics_"
                + d
                + "_"
                + m
                + "_"
                + str(number_of_features)
                + "_features_"
                + str(classi)
                + ".csv"
            )
