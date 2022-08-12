import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, zero_one_loss
from copy import deepcopy
from demv import DEMV
from fairlearn.metrics import MetricFrame

# METRICS

def disparate_impact(data_pred, group_condition, label_name, positive_label):
    unpriv_group_prob, priv_group_prob = _compute_probs(
        data_pred, label_name, positive_label, group_condition)
    return min(unpriv_group_prob / priv_group_prob,
               priv_group_prob / unpriv_group_prob) if unpriv_group_prob != 0 else \
        unpriv_group_prob / priv_group_prob


def statistical_parity(data_pred: pd.DataFrame, group_condition: dict, label_name: str, positive_label: str):
    query = '&'.join([f'{k}=={v}' for k, v in group_condition.items()])
    label_query = label_name+'=='+str(positive_label)
    unpriv_group_prob = (len(data_pred.query(query + '&' + label_query))
                         / len(data_pred.query(query)))
    priv_group_prob = (len(data_pred.query('~(' + query + ')&' + label_query))
                       / len(data_pred.query('~(' + query+')')))
    return unpriv_group_prob - priv_group_prob


def equalized_odds(data_pred: pd.DataFrame, group_condition: dict, label_name: str, positive_label: str):
    query = '&'.join([f'{k}=={v}' for k, v in group_condition.items()])
    label_query = label_name+'=='+str(positive_label)
    tpr_query = 'y_true == ' + str(positive_label)
    if(  len(data_pred.query(query + '&' + label_query)) == 0 ):
        unpriv_group_tpr = 0
    else:
        unpriv_group_tpr = (len(data_pred.query(query + '&' + label_query + '&' + tpr_query))
                            / len(data_pred.query(query + '&' + label_query)))


    if ( len(data_pred.query('~(' + query+')&' + label_query)) == 0  ):
        priv_group_tpr = 0
    else:
        priv_group_tpr = (len(data_pred.query('~(' + query + ')&' + label_query + '&' + tpr_query))
                        / len(data_pred.query('~(' + query+')&' + label_query)) )

    if (len(data_pred.query(query + '& ~(' + label_query + ')')) == 0 ):
        unpriv_group_fpr = 0
    else: 
        unpriv_group_fpr = (len(data_pred.query(query + '&' + label_query + '& ~(' + tpr_query + ')'))
                            / len(data_pred.query(query + '& ~(' + label_query + ')')))

    if ( len(data_pred.query('~(' + query+')& ~(' + label_query +')')) == 0):
        priv_group_fpr = 0
    else:
        priv_group_fpr = (len(data_pred.query('~(' + query + ')&' + label_query + '& ~(' + tpr_query + ')'))
                        / len(data_pred.query('~(' + query+')& ~(' + label_query +')')))

    return max ( np.abs(unpriv_group_tpr - priv_group_tpr) , np.abs(unpriv_group_fpr - priv_group_fpr) )


def _get_groups(data, label_name, positive_label, group_condition):
    query = '&'.join([str(k) + '==' + str(v)
                     for k, v in group_condition.items()])
    label_query = label_name + '==' + str(positive_label)
    unpriv_group = data.query(query)
    unpriv_group_pos = data.query(query + '&' + label_query)
    priv_group = data.query('~(' + query + ')')
    priv_group_pos = data.query('~(' + query + ')&' + label_query)
    return unpriv_group, unpriv_group_pos, priv_group, priv_group_pos


def _compute_probs(data_pred, label_name, positive_label, group_condition):
    unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = _get_groups(data_pred, label_name, positive_label,
                                                                             group_condition)
    unpriv_group_prob = (len(unpriv_group_pos)
                         / len(unpriv_group))
    priv_group_prob = (len(priv_group_pos)
                       / len(priv_group))
    return unpriv_group_prob, priv_group_prob


def _compute_tpr_fpr(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    FP = matrix.sum(axis=0) - np.diag(matrix)
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix.sum() - (FP + FN + TP)

    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    return FPR, TPR


def average_odds_difference(data_true: pd.DataFrame, data_pred: pd.DataFrame, group_condition: str, label: str):
    unpriv_group_true = data_true.query(group_condition)
    priv_group_true = data_true.drop(unpriv_group_true.index)
    unpriv_group_pred = data_pred.query(group_condition)
    priv_group_pred = data_pred.drop(unpriv_group_pred.index)

    y_true_unpriv = unpriv_group_true[label].values.ravel()
    y_pred_unpric = unpriv_group_pred[label].values.ravel()
    y_true_priv = priv_group_true[label].values.ravel()
    y_pred_priv = priv_group_pred[label].values.ravel()

    fpr_unpriv, tpr_unpriv = _compute_tpr_fpr(
        y_true_unpriv, y_pred_unpric)
    fpr_priv, tpr_priv = _compute_tpr_fpr(
        y_true_priv, y_pred_priv)
    return (fpr_unpriv - fpr_priv) + (tpr_unpriv - tpr_priv)/2


def zero_one_loss_diff(y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: list):
    mf = MetricFrame(metrics=zero_one_loss,
                     y_true=y_true,
                     y_pred=y_pred,
                     sensitive_features=sensitive_features)
    return mf.difference()


# TRAINING FUNCTIONS

def _train_test_split(df_train, df_test, label):
    x_train = df_train.drop(label, axis=1).values
    y_train = df_train[label].values.ravel()
    x_test = df_test.drop(label, axis=1).values
    y_test = df_test[label].values.ravel()
    return x_train, x_test, y_train, y_test




def cross_val(classifier, data, label, groups_condition, sensitive_features, positive_label, debiaser=None, exp=False, n_splits=10):
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    metrics = {
        'stat_par': [],
        'eq_odds' : [],
        'zero_one_loss': [],
        'disp_imp': [],
        'acc': [],
    }
    for train, test in fold.split(data):
        data = data.copy()
        df_train = data.iloc[train]
        df_test = data.iloc[test]
        model = deepcopy(classifier)
        if debiaser:
            run_metrics = _demv_training(model, debiaser, groups_condition, label,
                                         df_train, df_test, positive_label, sensitive_features)
        else:
            run_metrics = _model_train(df_train, df_test, label, model, defaultdict(
                list), groups_condition, sensitive_features, positive_label, exp)
        for k in metrics.keys():
            metrics[k].append(run_metrics[k])
    return model, metrics

def cross_val2(classifier, data, label, groups_condition, sensitive_features, positive_label, debiaser=None, exp=False, n_splits=10):
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    metrics = {
        'stat_par' : [],
        'eq_odds': [],
        'zero_one_loss': [],
        'disp_imp': [],
        'acc': []
    }
    pred = None
    for train, test in fold.split(data):
        data = data.copy()
        df_train = data.iloc[train]
        df_test = data.iloc[test]
        model = deepcopy(classifier)
        if debiaser:
            run_metrics = _demv_training(model, debiaser, groups_condition, label,
                                         df_train, df_test, positive_label, sensitive_features)
        else:
            run_metrics, predtemp = _model_train2(df_train, df_test, label, model, defaultdict(
                list), groups_condition, sensitive_features, positive_label, exp)
            pred = predtemp if pred is None else pred.append(predtemp)
        for k in metrics.keys():
            metrics[k].append(run_metrics[k])
    return model, metrics, pred

def cross_valbin(classifier, data, label, groups_condition, sensitive_features, positive_label, debiaser=None, exp=False, n_splits=10):
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    metrics = {
        'stat_par': [],
        'eq_odds': [],
        'zero_one_loss': [],
        'disp_imp': [],
        'acc': []
    }
    pred = None
    for train, test in fold.split(data):
        data = data.copy()
        df_train = data.iloc[train]
        df_test = data.iloc[test]
        model = deepcopy(classifier)
        if debiaser:
            run_metrics = _demv_training(model, debiaser, groups_condition, label,
                                         df_train, df_test, positive_label, sensitive_features)
        else:
            run_metrics, predtemp = _model_trainbin(df_train, df_test, label, model, defaultdict(
                list), groups_condition, sensitive_features, positive_label, exp)
            pred = predtemp if pred is None else pred.append(predtemp)
        for k in metrics.keys():
            metrics[k].append(run_metrics[k])
    return model, metrics, pred



def eval_demv(k, iters, data, classifier, label, groups, sensitive_features, positive_label=None, strategy='random'):
    ris = defaultdict(list)
    for i in range(0, iters+1, k):
        data = data.copy()
        demv = DEMV(1, debug=False, stop=i, strategy=strategy)
        _, metrics = cross_val(classifier, data, label, groups,
                               sensitive_features, debiaser=demv, positive_label=positive_label)
        #metrics = _compute_mean(metrics)
        ris['stop'].append(i)
        for k, v in metrics.items():
            val = []
            for i in v:
                val.append(np.mean(i))
            ris[k].append(val)
    return ris


def _demv_training(classifier, debiaser, groups_condition, label, df_train, df_test, positive_label, sensitive_features):
    metrics = defaultdict(list)
    for _ in range(30):
        df_copy = df_train.copy()
        data = debiaser.fit_transform(
            df_copy, [keys for keys in groups_condition.keys()], label)
        metrics = _model_train(data, df_test, label, classifier, metrics,
                               groups_condition, sensitive_features, positive_label)
    return metrics


def _model_train(df_train, df_test, label, classifier, metrics, groups_condition, sensitive_features, positive_label, exp=False):
    x_train, x_test, y_train, y_test = _train_test_split(
        df_train, df_test, label)
    model = deepcopy(classifier)
    model.fit(x_train, y_train,
              sensitive_features=df_train[sensitive_features]) if exp else model.fit(x_train, y_train)
    pred = model.predict(x_test)
    df_pred = df_test.copy()
    df_pred['y_true'] = df_pred[label]
    df_pred[label] = pred
    metrics['stat_par'].append(statistical_parity(
        df_pred, groups_condition, label, positive_label))
    metrics['eq_odds'].append(equalized_odds(
        df_pred, groups_condition, label, positive_label))
    metrics['disp_imp'].append(disparate_impact(
        df_pred, groups_condition, label, positive_label=positive_label))
    metrics['zero_one_loss'].append(zero_one_loss_diff(
        y_true=y_test, y_pred=pred, sensitive_features=df_test[sensitive_features].values))
    metrics['acc'].append(accuracy_score(y_test, pred))
    return metrics

def _model_train2(df_train, df_test, label, classifier, metrics, groups_condition, sensitive_features, positive_label, exp=False):
    x_train, x_test, y_train, y_test = _train_test_split(
        df_train, df_test, label)
    model = deepcopy(classifier)
    model.fit(x_train, y_train,
              sensitive_features=df_train[sensitive_features]) if exp else model.fit(x_train, y_train)
    pred = model.predict(x_test)
    df_pred = df_test.copy()
    df_pred['y_true'] = df_pred[label]
    df_pred[label] = pred

    df_pred.loc[:,"combined"] = 0
    tocomb = deepcopy(df_pred)

    for key,value in groups_condition.items():
        tocomb = df_pred.loc[ df_pred[key] == value ]
    
    df_pred.loc[ tocomb.index, 'combined' ] = 1

    df_pred = blackbox(df_pred, label)

    metrics['stat_par'].append(statistical_parity(  
        df_pred, groups_condition, label, positive_label))
    metrics['eq_odds'].append(equalized_odds(
        df_pred, groups_condition, label, positive_label))
    metrics['disp_imp'].append(disparate_impact(
        df_pred, groups_condition, label, positive_label=positive_label))
    metrics['zero_one_loss'].append(zero_one_loss_diff(
        y_true=y_test, y_pred=pred, sensitive_features=df_test[sensitive_features].values))
    metrics['acc'].append(accuracy_score(y_test, pred))
    return metrics, df_pred

def _model_trainbin(df_train, df_test, label, classifier, metrics, groups_condition, sensitive_features, positive_label, exp=False):
    x_train, x_test, y_train, y_test = _train_test_split(
        df_train, df_test, label)
    model = deepcopy(classifier)
    model.fit(x_train, y_train,
              sensitive_features=df_train[sensitive_features]) if exp else model.fit(x_train, y_train)
    pred = model.predict(x_test)
    df_pred = df_test.copy()
    df_pred['y_true'] = df_pred[label]
    df_pred[label] = pred

    df_pred.loc[:,"combined"] = 0
    tocomb = deepcopy(df_pred)

    for key,value in groups_condition.items():
        tocomb = df_pred.loc[ df_pred[key] == value ]
    
    df_pred.loc[ tocomb.index, 'combined' ] = 1


    df_pred = blackboxbin(df_pred, label)

    metrics['stat_par'].append(statistical_parity(
        df_pred, groups_condition, label, positive_label))
    metrics['eq_odds'].append(equalized_odds(
        df_pred, groups_condition, label, positive_label))
    metrics['disp_imp'].append(disparate_impact(
        df_pred, groups_condition, label, positive_label=positive_label))
    metrics['zero_one_loss'].append(zero_one_loss_diff(
        y_true=y_test, y_pred=pred, sensitive_features=df_test[sensitive_features].values))
    metrics['acc'].append(accuracy_score(y_test, pred))
    return metrics, df_pred


def print_metrics(metrics):
    print('Statistical parity: ', round(np.mean(
        metrics['stat_par']), 3), ' +- ', round(np.std(metrics['stat_par']), 3))
    print('Equalized Odds: ', round(np.mean(
        metrics['eq_odds']), 3), ' +- ', round(np.std(metrics['eq_odds']), 3))
    print('Disparate impact: ', round(np.mean(
        metrics['disp_imp']), 3), ' +- ', round(np.std(metrics['disp_imp']), 3))
    print('Zero one loss: ', round(np.mean(
        metrics['zero_one_loss']), 3), ' +- ', round(np.std(metrics['zero_one_loss']), 3))
    print('Accuracy score: ', round(np.mean(
        metrics['acc']), 3), ' +- ', round(np.std(metrics['acc']), 3))

def prepareplots(metrics, name):

    df = pd.DataFrame(metrics)
    columnlist = []
    for i in df.columns.values:
        if (i != 'stop'):
            columnlist.append(i)

    df = df.explode(columnlist)

    df.to_csv('ris/'+name+'_eval.csv')

    return df

def save_metrics(type, name, metric):
    df = pd.DataFrame(metric)
    df.explode(list(df.columns)).to_csv(f'ris/{name}_{type}.csv')


def blackbox(pred, label):
    from balancers import MulticlassBalancer

    pb = MulticlassBalancer(y = 'y_true', y_ = label, a = 'combined', data = pred)
    y_adj = pb.adjust(cv = True, summary = False)
    pred[label] = y_adj

    return pred


def blackboxbin(pred, label):
    from balancers import BinaryBalancer

    pb = BinaryBalancer(y = 'y_true', y_ = label, a = 'combined', data = pred)
    y_adj = pb.adjust(summary = False)
    pred[label] = y_adj

    return pred

def get_items(dataset,number_of_features):
    data = pd.read_csv("datarefactored/" + dataset + ".csv",index_col = 0)
    unpriv_group = {}
    sensitive_features = []
    sensfeat = pd.read_csv("datarefactored/sensitivefeatures.csv", index_col='dataset')
    for i in range(1,number_of_features+1):
        column = "unpriv_group" + str(i)
        string = sensfeat.loc[dataset,column]
        if( len(string.split(":")) == 2  ):
            key,value = string.split(":")
            
        else:
            key,value,threshold = string.split(":")
            threshold = int(threshold)
            data.loc[data[key] < threshold, key ] = 0
            data.loc[data[key] >= threshold, key ] = 1

        unpriv_group[key] = value
        sensitive_features.append(key)

    positive_label = sensfeat.loc[dataset,'positive_label']
    label = sensfeat.loc[dataset,'label']
    return data, unpriv_group, sensitive_features, label, positive_label
