# coding=utf-8
import re
import os
import time

import pandas as pd
from sqlalchemy import create_engine


root_dic = os.path.split(os.path.realpath(__file__))[0]
path = os.path.join(root_dic, 'data/data.h5')


class RawData:
    cache = {}

    root = 'data'
    source = {'essays': 'essays.csv',
              'outcomes': 'outcomes.csv',
              'projects': 'projects.csv',
              'resources': 'resources.csv',
              'donations': 'donations.csv'}
    read_opt = dict(true_values=['t'], false_values=['f'])

    def read(self, name, dtypes=None):
        t = self.source.get(name, None)
        if not t:
            raise NotImplementedError()
        path = os.path.join(self.root, t)

        rv = pd.read_csv(path, chunksize=50000, dtype=dtypes, **self.read_opt)

        print 'type checking'
        if dtypes is None:
            dtypes = {}
        rows = 0
        max_length = pd.DataFrame()

        chunk = rv.get_chunk(1)
        mem_usage = chunk.values.nbytes + chunk.index.nbytes
        col_type = {}

        for chunk in rv:
            # dtypes = dtypes.append(chunk.dtypes, ignore_index=True)
            rows += len(chunk)
            new_col_type = {r: set(v.unique()) for r, v in chunk.applymap(type).iteritems()}
            col_type = {k: col_type.get(k, set()) | v for k, v in new_col_type.items()}
        # print dtypes
        # dtypes = dtypes.max().to_dict()
        max_length = max_length.max().to_dict()

        bool_col = [k for k, v in col_type.items() if v == {bool, float}]
        print 'type checked'

        step = int(1200000 / mem_usage)
        rv = pd.read_csv(path, chunksize=step, dtype=dtypes, **self.read_opt)
        if hasattr(self, name) and callable(getattr(self, name)):
            rv = getattr(self, name)(rv)

        return rv, rows, max_length, bool_col

    def essays(self, essays):
        def clean(s):
            try:
                return " ".join(re.findall(r'\w+', s, flags=re.UNICODE | re.LOCALE)).lower()
            except:
                return " ".join(re.findall(r'\w+', "no_text", flags=re.UNICODE | re.LOCALE)).lower()

        for e in essays:
            e['essay'] = e['essay'].apply(clean)
            e['need_statement'] = e['need_statement'].apply(clean)
            e['essay_length'] = e.essay.apply(lambda x: len(str(x).split()))
            e['title_wc'] = e.title.apply(lambda x: len(str(x).split()))
            e['title_length'] = e.title.apply(lambda x: len(str(x)))
            yield e
        raise StopIteration()

    def outcomes(self, outcomes):
        donations = pd.read_csv('data/donations.csv')[['projectid', 'donor_acctid', 'payment_method']]
        donor_count = donations[['projectid', 'donor_acctid']].groupby('projectid').count()
        donor_count.columns = ['donor_count']

        green_donation = donations[donations['payment_method'].isin(['creditcard', 'paypal', 'amazon', 'check'])]
        green_donor_count = green_donation[['projectid', 'donor_acctid']].groupby('projectid').count()
        green_donor_count.columns = ['green_donor_count']

        for o in outcomes:
            o = o.join(donor_count, on='projectid', how='left')
            o = o.join(green_donor_count, on='projectid', how='left')
            o.donor_count = o.donor_count.fillna(0)
            o.green_donor_count = o.green_donor_count.fillna(0)
            yield o
        raise StopIteration()

    def projects(self, projects):
        for p in projects:
            p['date_posted'] = p['date_posted'].apply(pd.Timestamp)
            p['school_ncesid'] = p['school_ncesid'].apply(str)
            p['school_zip'] = p['school_zip'].apply(str)
            p['students_reached'] = p['students_reached'].apply(float)
            yield p
        raise StopIteration()

    def donations(self, donations):
        for d in donations:
            d['donation_timestamp'] = d['donation_timestamp'].apply(pd.Timestamp)
            yield d
        raise StopIteration()


def db():
    return create_engine('postgresql://localhost:5432/kdd')


def main(**kwargs):
    resource = RawData()
    con = db()

    for k, v in kwargs.items():
        print k
        if con.has_table(k):
            continue
        print('Start writing...')
        gen, total_rows, max_length, bool_col = resource.read(k)
        rows = 0
        for buf in gen:
            rows += len(buf)
            print '{}/{}'.format(rows, total_rows)
            buf.to_sql(k, con, if_exists='append', index=False, bool_columns=bool_col)
        print('Done!')

    con.connect()


if __name__ == '__main__':
    a = time.time()
    data_spec = {
        'projects': ['date_posted'],
        'outcomes': [''],
        'essays': ['projectid', 'essay_length', 'title_length'],
        'donations': ['projectid', 'donation_total'],
        'resources': ['projectid']
    }
    main(**data_spec)
    print time.time() - a
