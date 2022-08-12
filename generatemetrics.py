import sys
sys.path.append('../')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from fairlearn.reductions import ExponentiatedGradient, BoundedGroupLoss, ZeroOneLoss, GridSearch
import getdataset
from utils import *
from demv import DEMV
import warnings
warnings.filterwarnings('ignore')
import argparse


parser = argparse.ArgumentParser(description='Metrics generator for DEMV testing.')

parser.add_argument('dataset', type=str, 
                    help='Required argument: Chosen dataset to generate metrics for. Availability of datasets changes according to the chosen method.'
                    + ' All available datasets are: adult, cmc, compas, crime, drugs, german, obesity, park, wine.')

parser.add_argument('method', type=str, 
                    help='Required argument: Chosen method to generate metrics for. Can be biased, demv, eg, grid.')

parser.add_argument('number_of_features', type=int,
                    help='Required argument: Number of sensitive features in the dataset to consider, up to 4. ')

parser.add_argument('--classifier', type=str, nargs='?', default="logistic",
                    help='Optional argument: classifier to use. Can be logistic, svc, mlp, gradient. Defaults to Logistic Regression.')

args = parser.parse_args()

def run_metrics(method, data, unpriv_group, sensitive_features, label, positive_label,k):

    if(method == "biased" ):
        model, metr = cross_val(pipeline, data, label, unpriv_group, sensitive_features, positive_label=positive_label)
    elif(method == "eg"):
        constr = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
        exp = ExponentiatedGradient(pipeline, constr, sample_weight_name="classifier__sample_weight")
        model, metr = cross_val(exp, data.copy(), label, unpriv_group, sensitive_features, exp=True, positive_label=positive_label)
    elif(method == "grid"):
        constr = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
        exp = ExponentiatedGradient(pipeline, constr, sample_weight_name="classifier__sample_weight")
        grid = GridSearch(pipeline, constr, sample_weight_name="classifier__sample_weight")
        model, metr = cross_val(grid, data, label, unpriv_group, sensitive_features, positive_label, exp=True)
    else:
        demv = DEMV(round_level=1, strategy=method)
        model, metr = cross_val(pipeline, data.copy(), label, unpriv_group, sensitive_features, debiaser=demv, positive_label=positive_label)

    m = prepareplots( metr, 'discard' )

    return m


dataset = [args.dataset]
method = [args.method]
number_of_features = min(4,args.number_of_features)
if args.classifier is not None:
    classi = args.classifier
    if(classi == 'logistic'):
        classifier = LogisticRegression()
    elif(classi == 'gradient'):
        classifier = GradientBoostingRegressor()
    elif(classi == 'svc'):
        classifier = SVC()
    elif(classi == 'mlp'):
        classifier = MLPRegressor(activation = 'relu', solver='adam')
else:
    classi = 'logistic'
    classifier = LogisticRegression()

pipeline = Pipeline([
  ('scaler', StandardScaler()),
  ('classifier', classifier)
])


if dataset == ['all']:
    if number_of_features != 3:
        dataset = ['cmc','crime', 'drug', 'law', 'obesity', 'park', 'wine' ]
    else:
        dataset = ['cmc', 'crime', 'drug', 'law', 'wine']

if method == ['all']:    
    method = ['biased','demv', 'smote', 'adasyn', 'eg', 'grid']


for m in method:
    for d in dataset:
        print("Processing " + d  + " with " + m)

        data, label, positive_label, sensitive_features, unpriv_group, k = getdataset.getdataset(d, number_of_features)
        #else:
            #data, unpriv_group, sensitive_features, label, positive_label,k = get_items(d, number_of_features)

        result = run_metrics(m,data, unpriv_group, sensitive_features, label, positive_label,k)

        result.to_csv("ris/"+ str(number_of_features) + "features/metrics_" + d + "_" + m + "_" + str(number_of_features) + "_features_"+str(classi)+".csv")


