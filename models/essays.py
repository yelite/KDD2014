# coding=utf-8
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class EssaysTfidfModel:
    def __init__(self, projects, essays, outcomes,
                 train_id, test_id):
        self.data = pd.merge(projects, essays,
                             on='projectid', how='outer')
        self.data = pd.merge(self.data, outcomes,
                             on='projectid', how='outer')
        self.data = self.data[~pd.isnull(self.data.at_least_1_teacher_referred_donor)]

        self.train_id = train_id
        self.test_id = test_id

        self.vectorizer = TfidfVectorizer(min_df=4, max_features=800)
        self.lr_remains = LogisticRegression(class_weight='auto')
        self.lr_first = LogisticRegression(class_weight='auto')

        self.trained = False

    def train(self):
        data = self.data[self.data.projectid.isin(self.train_id)]
        self.vectorizer.fit(data.essay)
        self.data.tfidf = self.vectorizer.transform(self.data.essay)

        data = self.data[self.data.projectid.isin(self.train_id)]
        f_data = data.query("'2010-03-14' <= date_posted")
        self.lr_first.fit(f_data.tfidf, f_data.first_label)

        f_data = data.query("'2010-03-14' > date_posted")
        self.lr_remains.fit(f_data.tfidf, f_data.remains)

    def predict(self):
        data = self.data[self.data.projectid.isin(self.test_id)]
        p_remains = self.lr_remains.predict_proba(data.tfidf)
        p_first = self.lr_first.predict_proba(data.tfidf)

        rv = data.projectid
        rv.is_exciting = p_first*p_remains
        return rv