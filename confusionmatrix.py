from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from fairlearn.reductions import (BoundedGroupLoss, ExponentiatedGradient,
                                  GridSearch, ZeroOneLoss)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import getdataset
import utils
from demv import DEMV

pipeline = Pipeline([
  ('scaler', StandardScaler()),
  ('classifier', LogisticRegression())
])
classifier = pipeline


def plot_confusion_matrix(y_true, y_pred, classes, datasetname,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, method = 'Biased', sens = False):
                          
    if (sens):
        issens = "sensitive"
    else:
        issens = "not sensitive"

    if method is None:
        method = "Biased"
    if not title:
        if normalize:
            title = 'Normalized confusion matrix: ' + str(method) + "-" + str(issens)
        else:
            title = 'Confusion matrix, without normalization: ' + str(method) + "-" + str(issens)
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.grid(False)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]), 
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    
    plt.savefig('confusionmatrices/cm_sens=' + str(sens) + "_" + str(method) + "_" + str(datasetname) + "_normalized=" +str(normalize))
    return ax

def getprediction(data, label, classifier, groups_condition, sensitive_features, positive_label, debiaser = 'biased'):
    fold = KFold(n_splits = 10, shuffle = True)

    expbool = False

    if(debiaser == 'demv'):
        demv = DEMV(round_level=1, debug = False)
        debiaser = demv
        data = debiaser.fit_transform(
            data, [keys for keys in groups_condition.keys()], label)

    if(debiaser == 'eg'):
        constr = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
        exp = ExponentiatedGradient(pipeline, constr, sample_weight_name="classifier__sample_weight")
        expbool = True
        classifier = exp

    if(debiaser == 'grid'):
        constr = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
        exp = ExponentiatedGradient(pipeline, constr, sample_weight_name="classifier__sample_weight")
        grid = GridSearch(pipeline, constr, sample_weight_name="classifier__sample_weight")
        expbool = True
        classifier = grid
    
    newdata = deepcopy(data)
    newdata['y_true'] = data[label]

    for train, test in fold.split(data):

        model = deepcopy(classifier)

        df_train = data.iloc[train]
        df_test = data.iloc[test]

        x_train, x_test, y_train, y_test = utils._train_test_split(df_train, df_test, label)

        if expbool:
            model.fit(x_train, y_train, sensitive_features=df_train[sensitive_features]) 
        else:
            model.fit(x_train, y_train)

        pred = model.predict(x_test)
        newdata.iloc[test, newdata.columns.get_loc(label)] = pred

    return newdata
    
def generatecm(dataset, debiaser = None , normalize = False):
    data, label, positive_label, sensitive_features, unpriv_group, k = getdataset.getdataset(dataset, 2)
    newdata = getprediction(data, label, classifier, unpriv_group, sensitive_features, positive_label, debiaser=debiaser)
    classes = newdata['y_true'].unique()

    query = '&'.join([str(k) + '==' + str(v)
                     for k, v in unpriv_group.items()])
    datasens = newdata.query(query)
    plot_confusion_matrix(datasens['y_true'], datasens[label], classes, dataset, normalize = normalize, method = debiaser, sens = True)
    datanosens = newdata.query('~(' + query + ')')

    plot_confusion_matrix(datanosens['y_true'], datanosens[label], classes, dataset, normalize = normalize, method = debiaser, sens = False)

    #Invece del valore, nelle matrici, inserisci le label del dataset (Controlla le label originarie del law)
    #1. Mettiamo la label (classe di punteggi alta - media - bassa)
    #2. La label positiva la mettiamo in evidenza