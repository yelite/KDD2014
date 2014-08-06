# coding=utf-8
# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class ProjectAttrModel:
    features = [
        'school_charter',
        'school_magnet',
        'school_year_round',
        'school_nlns',
        'school_kipp',
        'school_charter_ready_promise',
        'teacher_prefix',
        'teacher_teach_for_america',
        'teacher_ny_teaching_fellow',
        'primary_focus_area',
        'resource_type',
        'poverty_level',
        'grade_level',
        'total_price_excluding_optional_support',
        'total_price_including_optional_support',
        'students_reached',
        'essay_length'
    ]

    def __init__(self, projects, outcomes, essays,
                 train_id, test_id):
        self.data = pd.merge(projects, outcomes,
                             on='projectid', how='outer')
        self.data = pd.merge(self.data, essays,
                             on='projectid', how='outer')

        filter_idx = pd.isnull(self.data.at_least_1_teacher_referred_donor)
        filter_idx &= self.data.projectid.isin(train_id)
        filter_idx &= self.data.date_posted >= '2010-03-14'
        filter_idx = ~filter_idx
        self.data = self.data[filter_idx]

        self.train_id = train_id
        self.test_id = test_id

        self.sgd_remains = LogisticRegression(penalty='l2',
                                              class_weight='auto')
        self.sgd_first = LogisticRegression(penalty='l2',
                                            class_weight='auto')

        self.vectorizer = DictVectorizer()
        self.tfidf = TfidfVectorizer(min_df=7, max_features=900)

        self.trained = False

    def train(self):
        data = self.data[self.data.projectid.isin(self.train_id)]
        self.vectorizer.fit(data[self.features].
                            fillna(0).
                            to_dict('record'))
        self.tfidf.fit(data.essay)

        f_data = data.query("'2010-03-14' <= date_posted")
        m = 0
        print '!'
        features = self.vectorizer.transform(f_data[self.features].
                                             fillna(m).
                                             to_dict('record')).toarray()
        tfidf = self.tfidf.transform(f_data.essay).toarray()
        features = np.column_stack((features, tfidf))

        self.sgd_first.fit(features,
                           np.array(f_data.first_label.apply(int)))

        f_data = data
        m = 0
        print '!'
        features = self.vectorizer.transform(f_data[self.features].
                                             fillna(m).
                                             to_dict('record')).toarray()
        tfidf = self.tfidf.transform(f_data.essay).toarray()
        features = np.column_stack((features, tfidf))

        self.sgd_remains.fit(features,
                             np.array(f_data.remains.apply(int)))

        self.trained = True

    def predict(self):
        if not self.trained:
            self.train()

        data = self.data[self.data.projectid.isin(self.test_id)]
        m = 0
        features = self.vectorizer.transform(data[self.features].
                                             fillna(m).
                                             to_dict('record')).toarray()
        tfidf = self.tfidf.transform(data.essay).toarray()
        features = np.column_stack((features, tfidf))

        p_remains = self.sgd_remains.predict_proba(features)
        p_first = self.sgd_first.predict_proba(features)

        rv = data[['projectid']]
        rv['is_exciting'] = p_remains[:, 1]*p_first[:, 1] + 0.25*(p_remains[:, 1]+p_first[:, 1])
        return rv