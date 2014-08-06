# coding=utf-8
import os
import time

from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, AdaBoostClassifier
from sklearn.preprocessing import normalize

from ._model import Model


def current_time():
    return time.strftime('%H:%M:%S', time.localtime())


root_dic = os.path.split(os.path.realpath(__file__))[0]

regular_features = [
    'school_latitude',
    'school_longitude',
    'school_state',
    'school_metro',
    # 'school_county',
    'school_charter',
    'school_magnet',
    'school_year_round',
    'school_nlns',
    'school_kipp',
    'school_charter_ready_promise',
    'teacher_prefix',
    'teacher_teach_for_america',
    'teacher_ny_teaching_fellow',
    'primary_focus_subject',
    'primary_focus_area',
    'secondary_focus_subject',
    'secondary_focus_area',
    'resource_type',
    'poverty_level',
    'grade_level',
    'fulfillment_labor_materials',
    'total_price_excluding_optional_support',
    'total_price_including_optional_support',
    'students_reached',
    'eligible_double_your_impact_match',
    'eligible_almost_home_match',
    'essay_length',
    'title_length',
    'title_wc',
    'eligible_double_your_impact_match',
    'eligible_almost_home_match',
    'time_inv'
]


class LabelTransformMixIn(object):
    _features = []
    _y_col = []

    def _process(self, data, resources, predict=False):
        vectorizer = resources['dictVectorizer']

        features = vectorizer.transform(data[self._features].
                                        fillna(0).
                                        to_dict('record'))
        y = data[self._y_col]
        if not predict:
            weight_p, weight_n = 1.0/sum(y==True), 1.0/sum(y==False)
            print weight_n, weight_p
            weights = y.apply(lambda x: weight_p if x else weight_n)
        else:
            y = np.array([])
            weights = []

        y = (y, weights)
        return features, y


class NaiveTransformMixIn(object):
    _features = []
    _y_col = None

    def _process(self, data, resources):
        vectorizer = resources['dictVectorizer']
        tfidf = resources['tfidf']

        features = vectorizer.transform(data[self._features].
                                        fillna(0).
                                        to_dict('record'))
        essay_features = tfidf.transform(data.essay)

        features = hstack([features, essay_features])
        if self._y_col in data.columns:
            y = data[self._y_col]
        else:
            y = np.array([])
        return features, y


class DenseNaiveTransformMixIn(NaiveTransformMixIn):
    def _process(self, data, resources):
        X, y = super(DenseNaiveTransformMixIn, self)._process(data, resources)
        return X.toarray(), y


class DenseBalancedLabelTransformMixIn(LabelTransformMixIn):
    def _process(self, data, resources, predict=False):
        X, y = super(DenseBalancedLabelTransformMixIn, self)._process(data, resources, predict)
        return X.toarray(), y


class FloatNaiveTrans(NaiveTransformMixIn):
    def _process(self, data, resources):
        features, y = super(FloatNaiveTrans, self)._process(data, resources)
        return features.astype(np.float32), y.astype(np.float32)


class EssayTransformMixIn:
    _features = []
    _y_col = None

    def _process(self, data, resources):
        tfidf = resources['tfidf']
        features = tfidf.transform(data.essay)
        if self._y_col in data.columns:
            y = data[self._y_col]
        else:
            y = np.array([])
        return features, y


class TwoLabelClassifier(Model):
    _features = []
    _y_col = None

    def _train(self, train_data, resources):
        self.classifier = self._model

        filter_idx = ~pd.isnull(train_data[self._y_col])
        train_data = train_data[filter_idx]

        features, y = self._process(train_data, resources)
        y, weight = y
        print features.shape

        print('Begin Training - {}'.format(current_time()))
        self.classifier.fit(features, np.array(y),
                            sample_weight=np.array(weight))
        return train_data, resources

    def _predict(self, test_data, resources, results):
        features, _ = self._process(test_data, resources, predict=True)
        # features = self.selector.transform(features)
        return self.classifier.predict_proba(features)[:, 1]

    @property
    def _y_label(self):
        return self._y_col

    @property
    def _model(self):
        raise NotImplementedError


class ScalarRegressor(Model):
    _features = []
    _y_col = None
    _y_label = None

    def _train(self, train_data, resources):
        self.regressor = self._model

        filter_idx = ~pd.isnull(train_data[self._y_col])
        train_data = train_data[filter_idx]

        features, y = self._process(train_data, resources)
        print features.shape

        print('Begin Training - {}'.format(current_time()))
        self.regressor.fit(features, np.array(y))
        y_pred = self.regressor.predict(features)
        self.error = mean_squared_error(y, y_pred) ** 0.5
        return train_data, resources

    def _predict(self, test_data, resources, results):
        features, _ = self._process(test_data, resources, predict=True)
        # features = self.selector.transform(features)
        y_pred = self.regressor.predict(features)
        y_label = self._val_to_label(y_pred)
        return y_label

    def _val_to_label(self, value):
        raise NotImplementedError

    @property
    def _model(self):
        raise NotImplementedError


class ClassicPredictor(TwoLabelClassifier, DenseBalancedLabelTransformMixIn):
    @property
    def _model(self):
        return BaggingClassifier(
            base_estimator=DecisionTreeClassifier(
                min_weight_fraction_leaf=0.0035),
            n_estimators=120,
            verbose=True,
            n_jobs=-1,
            max_samples=0.15,
            max_features=0.3
        )
        # return lm.LogisticRegression(class_weight='auto', fit_intercept=False,
        #                              C=200)

class FundedPredictor(ClassicPredictor):
    _features = regular_features
    _y_col = 'fully_funded'
    _y_label = _y_col


class ThoughtfulDonorPredictor(ClassicPredictor):
    _features = regular_features
    _y_col = 'donation_from_thoughtful_donor'
    _y_label = _y_col


class TeacherReferredDonorPredictor(ClassicPredictor):
    _features = regular_features
    _y_label = 'at_least_1_teacher_referred_donor'
    # _y_col = 'teacher_referred_count'
    _y_col = _y_label


class GreenRatioPredictor(ClassicPredictor):
    _features = regular_features
    # _y_col = 'green_donor_count'
    _y_label = 'at_least_1_green_donation'
    _y_col = _y_label


class DonationPredictor(ClassicPredictor):
    _features = regular_features
    _y_col = 'one_non_teacher_referred_donor_giving_100_plus'
    _y_label = _y_col


class NonTDonorPredictor(ClassicPredictor):
    _features = regular_features
    # _y_col = 'non_teacher_referred_count'
    _y_label = 'three_or_more_non_teacher_referred_donors'
    _y_col = _y_label


class GreetRatioPredictor(ClassicPredictor):
    _features = regular_features
    # _y_col = 'great_messages_proportion'
    _y_label = 'great_chat'
    _y_col = _y_label


class ExcitingPredictor(ClassicPredictor):
    _features = regular_features
    # _y_col = 'great_messages_proportion'
    _y_label = 'is_exciting'
    _y_col = _y_label


all_subs = [FundedPredictor, DonationPredictor, ThoughtfulDonorPredictor,
            GreetRatioPredictor, GreenRatioPredictor,
            NonTDonorPredictor, TeacherReferredDonorPredictor]


class DecompositionModel(Model):
    _sub_models = all_subs
    features = regular_features

    def _train(self, train_data, resources):
        sample_length = len(train_data)
        dict_status_path = os.path.join(root_dic,
                                        'dict_vectorizer_{}.status'.
                                        format(sample_length))
        if os.path.isfile(dict_status_path):
            dictVectorizer = joblib.load(dict_status_path)
        else:
            dictVectorizer = DictVectorizer()
            dictVectorizer.fit(train_data[self.features].
                               fillna(0).
                               to_dict('record'))
            joblib.dump(dictVectorizer, dict_status_path)

        tfidf_status_path = os.path.join(root_dic,
                                         'tfidf_vectorizer_{}.status'.
                                         format(sample_length))
        if os.path.isfile(tfidf_status_path):
            tfidf = joblib.load(tfidf_status_path)
        else:
            tfidf = TfidfVectorizer(min_df=40, max_features=300)
            tfidf.fit(train_data.essay)
            joblib.dump(tfidf, tfidf_status_path)

        resources['dictVectorizer'] = dictVectorizer
        resources['tfidf'] = tfidf
        print 'Head Processing Completed'
        return train_data, resources

    def scale(self, X):
        min = np.min(X)
        max = np.max(X)
        return (X - min) / (max - min)

    def _predict(self, data, resources, results):
        rv = pd.DataFrame(dict(projectid=data.iloc[:, 0]))
        for k in self.predictors:
            rv[k._y_label] = results[k.name]
        # joblib.dump(rv, 'data.db'.format(current_time()))

        k1 = [
            'three_or_more_non_teacher_referred_donors',
            'one_non_teacher_referred_donor_giving_100_plus',
            'donation_from_thoughtful_donor'
        ]

        k2 = [
            'fully_funded',
            'great_chat',
            'at_least_1_green_donation',
            'at_least_1_teacher_referred_donor'
        ]

        k3 = k1 + k2

        # rv['at_least_1_green_donation'] = 1 - rv['at_least_1_green_donation']

        for k in k3:
            rv[k] = self.scale(rv[k])

        p = 1
        for k in k1:
            p *= (1 - rv[k])
        p = 1 - p

        for k in k2:
            p *= rv[k]
        p = (p - np.min(p)) / (np.max(p) - np.min(p))

        rv['is_exciting'] = p
        rv = rv[['projectid', 'is_exciting']]
        return rv

    def validate_sub(self, sub_model, train_data, test_data):
        self._sub_models = [sub_model]
        self.predictors = [P(parent=self)
                           for P in self._sub_models]
        self.train(train_data)
        rv = test_data[['projectid']]
        rv[self.predictors[0].name] = self.predictors[0].predict(test_data)
        return rv



