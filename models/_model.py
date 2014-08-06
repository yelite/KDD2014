# coding=utf-8

import os
import time
root_dic = os.path.split(os.path.realpath(__file__))[0]
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class NotTrainedError(RuntimeError):
    pass


def current_time():
    return time.strftime('%H:%M:%S', time.localtime())


class Model(object):
    _sub_models = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.trained = False
        self.resources = None
        self.predictors = [P(**kwargs)
                           for P in self._sub_models]

    def _train(self, train_data, resources):
        raise NotImplementedError()

    def train(self, train_data, resources=None):
        if resources is None:
            resources = {}
        print('Training {} - {}'.format(self.__class__.__name__, current_time()))
        train_data, self.resources = self._train(train_data, resources)
        map(lambda x: x.train(train_data,
                              self.resources),
            self.predictors)
        print('Trained - {}'.format(current_time()))
        self.trained = True

    def _predict(self, test_data, resources, results):
        raise NotImplementedError()

    def predict(self, test_data, results=None):
        if not self.trained:
            raise NotTrainedError()
        results = {k.name: k.predict(test_data)
                   for k in self.predictors}
        return self._predict(test_data,
                             resources=self.resources,
                             results=results)

    @property
    def name(self):
        return self.__getattribute__('_y_col') \
               or self.__getattribute__('_name') \
               or self.__class__.__name__
