import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, zero_one_loss
from copy import deepcopy
from demv import DEMV


from fairlearn.metrics import MetricFrame

import matplotlib.pyplot as plt
import seaborn as sns


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


# PLOT FUNCTIONS


def plot_metrics_curves(df, points=None, title='', ax=None):
    """Plot DEMV iteration curves"""

    metrics = {'stat_par': 'Statistical Parity', 'eq_odds':'Equalized Odds','zero_one_loss': 'Zero One Loss',
               'disp_imp': 'Disparate Impact', 'acc': 'Accuracy'}
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 8))
    for k, v in metrics.items():
        sns.lineplot(data=df, y=k, x='stop', label=v, ci='sd', ax=ax)
    if points is not None:
        for k, v in points.items():
            ax.plot(v['x'], v['y'], v['type'], label=k, markersize=10)
    ax.set(ylabel='Value', xlabel='Stop value')
    ax.lines[0].set_linestyle("--")
    ax.lines[0].set_marker('o')
    #lines[1] is zero_one_loss
    ax.lines[1].set_marker('x')
    ax.lines[1].set_markeredgecolor('orange')
    ax.lines[1].set_linestyle("--")

    ax.lines[2].set_marker('+')
    ax.lines[2].set_markeredgecolor('green')
    ax.lines[2].set_linestyle(":")
    ax.lines[2].set_markevery(0.001)

    ax.lines[3].set_color("black")
    ax.legend(handlelength=5, loc="upper center", bbox_to_anchor=(
        0.5, -0.03), ncol=3, fancybox=True, shadow=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_grid(dfs, ys, iter, types, metrics):
    fig = plt.figure(dpi=60, tight_layout=True)
    fig.set_size_inches(15, 5, forward=True)

    gs = fig.add_gridspec(1, len(dfs))

    ax = np.zeros(3, dtype=object)

    for k, v in dfs.items():
        df = v
        points = ys[k]
        iters = iter[k]
        i = list(dfs.keys()).index(k)
        ax[i] = fig.add_subplot(gs[0, i])
        for key, v in metrics.items():
            ax[i] = sns.lineplot(data=df, y=key, x='stop', label=v, ci='sd')

        for key, v in points.items():
            ax[i].plot(iters, points[key], types[key],
                       label=key, markersize=10)

        ax[i].set_ylabel('Value', fontsize=15)
        ax[i].set_xlabel('Iterations', fontsize=15)
        ax[i].set_title(k, fontsize=15)

        ax[i].lines[0].set_linestyle("--")
        ax[i].lines[0].set_marker('o')
        #lines[1] is zero_one_loss
        ax[i].lines[1].set_marker('x')
        ax[i].lines[1].set_markeredgecolor('orange')
        ax[i].lines[1].set_linestyle("--")

        ax[i].lines[2].set_marker('+')
        ax[i].lines[2].set_markeredgecolor('green')
        ax[i].lines[2].set_linestyle(":")
        ax[i].lines[2].set_markevery(0.001)
        ax[i].get_legend().remove()
        ax[i].plot()

    handles, labels = ax[len(dfs)-1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, -0.03), ncol=4, prop={'size': 15}, fancybox=True, shadow=True)
    fig.savefig('../img/Grid.pdf', bbox_extra_artists=(lgd,),
                bbox_inches='tight')


def compare_curves(df_rand, df_smote, title1='', title2=''):
    """Compare Random DEMV and Smote DEMV curves"""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    metrics = {'stat_par': 'Statistical Parity', 'eq_odds': 'Equalized Odds', 'zero_one_loss': 'Zero One Loss',
               'disp_imp': 'Disparate Impact', 'acc': 'Accuracy'}
    for k, v in metrics.items():
        sns.lineplot(data=df_rand, y=k, x='stop', label=v,
                     ci='sd', ax=axs[0], legend=None)
        sns.lineplot(data=df_smote, y=k, x='stop', label=v,
                     ci='sd', ax=axs[1], legend=None)
    for ax in axs:
        ax.set(ylabel='Value', xlabel='Iterations')
        ax.lines[0].set_linestyle("--")
        ax.lines[0].set_marker('o')
        #lines[1] is zero_one_loss
        ax.lines[1].set_marker('x')
        ax.lines[1].set_markeredgecolor('orange')
        ax.lines[1].set_linestyle("--")

        ax.lines[2].set_marker('+')
        ax.lines[2].set_markeredgecolor('green')
        ax.lines[2].set_linestyle(":")
        ax.lines[2].set_markevery(0.001)

        ax.lines[3].set_marker('v')
        ax.lines[3].set_linestyle(":")
    axs[0].set(title=title1)
    axs[1].set(title=title2)
    handles, labels = axs[2-1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, -0.03), ncol=4, prop={'size': 15}, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig('curves.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def plot_gridmulti(dfs, ys, iter, types, metrics, name='GridMulti'):
    fig = plt.figure(dpi=60, tight_layout=True)
    fig.set_size_inches(15, 8, forward=True)

    gs = fig.add_gridspec(2, 6)
    ax = np.zeros(5, dtype=object)

    for k, v in dfs.items():

        df = v
        points = ys[k]
        iters = iter[k]
        i = list(dfs.keys()).index(k)

        if(i == 0):
            ax[i] = fig.add_subplot(gs[0, :2])
        elif(i == 1):
            ax[i] = fig.add_subplot(gs[0, 2:4])
        elif(i == 2):
            ax[i] = fig.add_subplot(gs[0, 4:])
        elif(i == 3):
            ax[i] = fig.add_subplot(gs[1, 1:3])
        elif(i == 4):
            ax[i] = fig.add_subplot(gs[1, 3:5])

        for key, v in metrics.items():
            ax[i] = sns.lineplot(data=df, y=key, x='stop', label=v, ci='sd')

        for key, v in points.items():
            ax[i].plot(iters, points[key], types[key],
                       label=key, markersize=10)

        ax[i].set_ylabel('Value', fontsize=15)
        ax[i].set_xlabel('Iterations', fontsize=15)
        ax[i].set_title(k, fontsize=15)

        ax[i].lines[0].set_linestyle("--")
        ax[i].lines[0].set_marker('o')
        #lines[1] is zero_one_loss
        ax[i].lines[1].set_marker('x')
        ax[i].lines[1].set_markeredgecolor('orange')
        ax[i].lines[1].set_linestyle("--")

        ax[i].lines[2].set_marker('+')
        ax[i].lines[2].set_markeredgecolor('green')
        ax[i].lines[2].set_linestyle(":")
        ax[i].lines[2].set_markevery(0.001)
        ax[i].get_legend().remove()
        ax[i].plot()

    handles, labels = ax[len(dfs)-1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, -0.03), ncol=4, prop={'size': 15}, fancybox=True, shadow=True)
    fig.savefig(f'../img/{name}.pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')


def preparepoints(metrics, iters):

    types = {'Stastical Parity (Exp Gradient)': 'xb',
            'Equalized Odds (Exp Gradient)' : 'ob',
             'Zero One Loss (Exp Gradient)': 'xy',
             'Disparate Impact (Exp Gradient)': 'xg',
             'Accuracy (Exp Gradient)': 'xr',
             }

    rename = {'Stastical Parity (Exp Gradient)': 'stat_par',
               'Equalized Odds (Exp Gradient)' : 'eq_odds', 
              'Zero One Loss (Exp Gradient)': 'zero_one_loss',
              'Disparate Impact (Exp Gradient)': 'disp_imp',
              'Accuracy (Exp Gradient)': 'acc'
              }

    points = {}

    for k in types.keys():
        points[k] = {'x': iters, 'y': np.mean(
            metrics[rename[k]]), 'type':  types[k]}

    return points


def unprivpergentage(data, unpriv_group, iters):
    unprivdata = data.copy()
    for k, v in unpriv_group.items():
        unprivdata = unprivdata[(unprivdata[k] == v)]

    xshape, _ = unprivdata.shape

    print('Dataset size:', data.shape[0])
    print('Unprivileged group size:', xshape)
    print('Percentage of unprivileged group:', (xshape/data.shape[0])*100)
    print('Number of iterations:', iters)


def prepareplots(metrics, name):

    df = pd.DataFrame(metrics)
    columnlist = []
    for i in df.columns.values:
        if (i != 'stop'):
            columnlist.append(i)

    df = df.explode(columnlist)

    df.to_csv('ris/'+name+'_eval.csv')

    return df


def gridcomparison(dfs, dfsm, ys, ysm, iter, iterm, types, metrics):

    fig = plt.figure(dpi=60, tight_layout=True)
    fig.set_size_inches(15, 15, forward=True)

    gs = fig.add_gridspec(5, 2)
    ax = np.zeros(10, dtype=object)

    for k, v in dfs.items():
        df = v
        points = ys[k]
        iters = iter[k]
        i = list(dfs.keys()).index(k)

        if(i == 0):
            ax[i] = fig.add_subplot(gs[0, 0])
        elif(i == 1):
            ax[i] = fig.add_subplot(gs[1, 0])
        elif(i == 2):
            ax[i] = fig.add_subplot(gs[2, 0])
        elif(i == 3):
            ax[i] = fig.add_subplot(gs[3, 0])
        elif(i == 4):
            ax[i] = fig.add_subplot(gs[4, 0])

        for key, v in metrics.items():
            ax[i] = sns.lineplot(data=df, y=key, x='stop', label=v, )

        for key, v in points.items():
            ax[i].plot(iters, points[key], types[key],
                       label=key, markersize=10)

        ax[i].set(ylabel='Value', xlabel='Stop value')
        ax[i].set_title(k + " single var")

        ax[i].lines[0].set_linestyle("--")
        ax[i].lines[0].set_marker('o')
        #lines[1] is zero_one_loss
        ax[i].lines[1].set_marker('x')
        ax[i].lines[1].set_markeredgecolor('orange')
        ax[i].lines[1].set_linestyle("--")

        ax[i].lines[2].set_marker('+')
        ax[i].lines[2].set_markeredgecolor('green')
        ax[i].lines[2].set_linestyle(":")
        ax[i].lines[2].set_markevery(0.001)
        ax[i].get_legend().remove()
        ax[i].plot()

    for k, v in dfsm.items():
        df = v
        points = ysm[k]
        iters = iterm[k]
        i = list(dfsm.keys()).index(k)

        if(i == 0):
            ax[i] = fig.add_subplot(gs[0, 1])
        elif(i == 1):
            ax[i] = fig.add_subplot(gs[1, 1])
        elif(i == 2):
            ax[i] = fig.add_subplot(gs[2, 1])
        elif(i == 3):
            ax[i] = fig.add_subplot(gs[3, 1])
        elif(i == 4):
            ax[i] = fig.add_subplot(gs[4, 1])

        for key, v in metrics.items():
            ax[i] = sns.lineplot(data=df, y=key, x='stop', label=v, )

        for key, v in points.items():
            ax[i].plot(iters, points[key], types[key],
                       label=key, markersize=10)

        ax[i].set(ylabel='Value', xlabel='Stop value')
        ax[i].set_title(k)

        ax[i].lines[0].set_linestyle("--")
        ax[i].lines[0].set_marker('o')
        #lines[1] is zero_one_loss
        ax[i].lines[1].set_marker('x')
        ax[i].lines[1].set_markeredgecolor('orange')
        ax[i].lines[1].set_linestyle("--")

        ax[i].lines[2].set_marker('+')
        ax[i].lines[2].set_markeredgecolor('green')
        ax[i].lines[2].set_linestyle(":")
        ax[i].lines[2].set_markevery(0.001)
        ax[i].get_legend().remove()
        ax[i].plot()

    handles, labels = ax[len(dfs)-1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, -0.03), ncol=4, prop={'size': 15}, fancybox=True, shadow=True)
    fig.savefig('img/GridMultiSingleVar.pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')


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

def plot_baselines_comparison(plot_data, figname, palette):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig_data = plot_data[(plot_data['metric'] == 'Statistical Parity') | (
        plot_data['metric'] == 'Equalized Odds') | (plot_data['metric'] == 'Zero One Loss')]
    sns.barplot(data=fig_data, x='metric', y='value',
                hue='type', ci='sd',  ax=ax[0],palette = palette )
    ax[0].set(xlabel='', ylabel='', title='Metrics whose optimal value is zero')
    ax[0].get_legend().remove()
    fig_data = plot_data[(plot_data['metric'] == 'Disparate Impact') | (
        plot_data['metric'] == 'Accuracy')]
    sns.barplot(data=fig_data, x='metric', y='value',
                hue='type', ci='sd', ax=ax[1], palette = palette)
    ax[1].set(xlabel='', ylabel='', title='Metrics whose optimal value is one',)
    ax[1].get_legend().remove()
    handles, labels = ax[1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(
        0.5, -0.05),
        ncol=5, prop={'size': 10}, fancybox=True, shadow=True, title='Methods')
    plt.tight_layout()
    plt.savefig(f'imgs/{figname}.pdf',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
