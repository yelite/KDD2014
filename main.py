# coding=utf-8
import time

import models.decomposition as md
import pandas as pd
from resource import db

def extract_data():
    con = db()

    train_criteria = ['2011-02-01', '2014-01-01']
    test_criteria = ['2014-01-01', '2015-01-01']
    query = 'SELECT * FROM projects ' \
            'LEFT OUTER JOIN outcomes ' \
            'USING(projectid) ' \
            ' JOIN essays ' \
            'USING(projectid) ' \
            'WHERE date_posted >= %s AND date_posted < %s'
    train_data = pd.read_sql_query(query, con, params=train_criteria)
    test_data = pd.read_sql_query(query, con, params=test_criteria)
    print 'Data Fetched'
    print "Train: {}".format(len(train_data))
    print "Test: {}".format(len(test_data))
    return test_data, train_data


test_data, train_data = extract_data()

model = md.DecompositionModel()

# model.train(train_data)

rv = model.validate_sub(md.ExcitingPredictor, train_data, test_data)
# rv = model.predict(test_data)

timestamp = time.strftime('%m_%d_%H_%M_%S', time.localtime())
rv.to_csv('output/p_{}.csv'.format(timestamp), index=False)