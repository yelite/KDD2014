# coding=utf-8

import logging

import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import models.decomposition as md
from resource import db
import numpy as np

logging.basicConfig(filename='test.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
import pylab as pl


def calc_roc(test_data, rv, sub_model):
    y = test_data[sub_model._y_label]
    fpr, tpr, thresholds = metrics.roc_curve(y,
                                             rv[sub_model._y_col],
                                             pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print roc_auc
    logging.info(': '.join((sub_model.__name__, str(roc_auc))))
    return fpr, roc_auc, tpr


def plot_roc(fpr, tpr, roc_auc):
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()


def extract_data():
    con = db()

    train_criteria = ['2009-01-01', '2013-01-01']
    test_criteria = ['2013-01-01', '2014-01-01']
    query = 'SELECT * FROM projects ' \
            'LEFT OUTER JOIN outcomes ' \
            'USING(projectid) ' \
            'JOIN essays ' \
            'USING(projectid) ' \
            'WHERE date_posted >= %s AND date_posted < %s'
    train_data = pd.read_sql_query(query, con, params=train_criteria)
    test_data = pd.read_sql_query(query, con, params=test_criteria)
    print 'Data Fetched'
    print "Train: {}".format(len(train_data))
    print "Test: {}".format(len(test_data))
    return test_data, train_data


def test_sub_roc(sub_model, model, plot=False):
    test_data, train_data = extract_data()

    rv = model.validate_sub(sub_model, train_data, test_data)

    fpr, roc_auc, tpr = calc_roc(test_data, rv, sub_model)
    if plot:
        plot_roc(fpr, tpr, roc_auc)


def test_sub_error(sub_model, model):
    test_data, train_data = extract_data()

    rv = model.validate_sub(sub_model, train_data, test_data)

    y_real = test_data[sub_model._y_col]
    y_real = y_real.fillna(y_real.mean())
    print metrics.mean_squared_error(y_real,
                                     rv[sub_model._y_col])


def test_roc(model, plot=False):
    test_data, train_data = extract_data()
    model.train(train_data)
    rv = model.predict(test_data)

    y = test_data.is_exciting
    fpr, tpr, thresholds = metrics.roc_curve(y,
                                             rv.is_exciting,
                                             pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print roc_auc
    logging.info(str(roc_auc))

    plot_roc(fpr, tpr, roc_auc)


def scale(X):
    min = np.min(X)
    max = np.max(X)
    return (X - min) / (max - min)


def test_remains():
    from sklearn.externals import joblib

    k1 = [
        'three_or_more_non_teacher_referred_donors',
        'one_non_teacher_referred_donor_giving_100_plus',
        # 'donation_from_thoughtful_donor'
    ]

    k2 = [
        # 'fully_funded',
        'great_chat',
        # 'at_least_1_green_donation',
        'at_least_1_teacher_referred_donor'
    ]

    k3 = k1 + k2

    test_data = joblib.load('test_data')
    # test_data, train_data = extract_data()
    # test_data = test_data.sort('projectid')
    # assert (test_data.at_least_1_green_donation[test_data.green_donor_count<1]).all()

    results = joblib.load('data.db')
    results = results.sort('projectid')
    # print rv.keys()

    # results['at_least_1_green_donation'] = 1 - results['at_least_1_green_donation']

    for k in k3:
        results[k] = scale(results[k])
    p = 1
    for k in k1:
        p *= (1 - results[k])
    p = 1 - p

    for k in k2:
        p *= results[k]
    p = (p - np.min(p)) / (np.max(p) - np.min(p))
    # p = normalize(np.array(p), axis=1)

    # for k in k2:
    # p += 0.2 * results[k]

    # for k in k3:
    #     print ''
    #     print k
    #     y = test_data[k]
    #     fpr, tpr, thresholds = metrics.roc_curve(y,
    #                                              results[k],
    #                                              pos_label=1)
    #     roc_auc = metrics.auc(fpr, tpr)
    #     print roc_auc
    #     logging.info(str(roc_auc))

    y = test_data.is_exciting
    fpr, tpr, thresholds = metrics.roc_curve(y,
                                             p,
                                             pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print roc_auc

    # plot_roc(fpr, tpr, roc_auc)


if __name__ == '__main__':
    model = md.DecompositionModel()
    # test_roc(model, True)
    # test_sub_roc(md.DonationPredictor, model)
    # test_remains()
    test_sub_roc(md.ExcitingPredictor, model)
    # test_sub_roc(md.TeacherReferredDonorPredictor, model)