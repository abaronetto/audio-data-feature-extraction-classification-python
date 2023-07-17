import copy
import os
import sys
from pathlib import Path

import pandas as pd
import torch.nn

path = Path(os.getcwd())
sys.path.append(os.getcwd() + '/../')

import numpy as np

def save_json(data,fn: str):
    import json
    with open(fn, 'w') as f:
        json.dump(data, f, indent=4)

def resample(X, y, method='random'):
    import six
    import sklearn.neighbors._base
    import pandas as pd
    sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
    sys.modules['sklearn.externals.six'] = six
    if method == 'random':
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler()
    elif method =='SMOTE':
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(n_jobs=-1, sampling_strategy='all', k_neighbors=5)
    if isinstance(X, pd.DataFrame):
        col_names = X.columns
        X, y = sampler.fit_resample(X, y)
        X = pd.DataFrame(data=X, columns=col_names)
        if 'label' in col_names:
            X['label'] = y
        else:
            X.insert(0, 'label', y)
    else:
        X, y = sampler.fit_resample(X, y)
    return X, y


class DataClassifier:
    def __init__(self, data, label, features, cv_method='fixed', standardize=True, groups=None, train_data=None, train_label=None, remove_lower_outliers=False, means_init=False):
        # Runs classification (using defined cross-validation) using any scikit-learn classifier.
        self.cv_method = cv_method
        self.features = features
        self.data = data
        self.label = label
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.clf = None
        # only for GMM
        self.remove_lower_outliers = remove_lower_outliers
        self.means_init = means_init

        if train_data is not None:
            self.X_train = train_data

        if cv_method == 'fixed':
            self.recognised = None
            self.insertions = None
            self.relevant = None
            # Fixed cross-validation
            # Split the dataset in 70% train set and 30% test set
            if train_data is not None:
                self.y_train = train_label
                self.X_test, self.y_test = data, label
            else:
                from sklearn.model_selection import train_test_split
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, label, test_size=0.3)
                self.train_size = self.X_train.shape[0]
                self.test_size = self.X_test.shape[0]

        elif cv_method == 'lopo':
            assert train_data is None, "The current implemented cv method does not support a different training set"
            from sklearn.model_selection import LeaveOneGroupOut
            # Leave One Participant Out
            self.groups = groups
            self.logo = LeaveOneGroupOut()
            self.y_pred = {}
            self.gt = {}
            self.train_size = {}
            self.test_size = {}

        elif 'fold' in cv_method:
            assert train_data is None, "The current implemented cv method does not support a different training set"
            from sklearn.model_selection import KFold
            self.kf = KFold(n_splits=int(cv_method[:cv_method.find('fold')]))
            self.y_pred = {}
            self.gt = {}
            self.train_size = {}
            self.test_size = {}

        self.standardize = standardize
        if self.standardize:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        self.precision = None
        self.recall = None
        self.params = 'default'

    def run_classifier(self, clf, **params):
        # not for GMM
        self.clf = clf(**params)

        if self.cv_method == 'fixed':

            if self.standardize:
                self.X_train = self.scaler.fit_transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)

            self.clf.fit(self.X_train, self.y_train)
            self.y_pred = self.clf.predict(self.X_test)

        elif self.cv_method == 'lopo':
            self.precision = {}
            self.recall = {}
            i = 1
            for train, test in self.logo.split(self.data, self.label, self.groups):
                if self.standardize:
                    self.data[train] = self.scaler.fit_transform(self.data[train])
                    self.data[test] = self.scaler.transform(self.data[test])

                self.clf.fit(self.data[train], self.label[train])
                self.y_pred[f"lopo_with_{int(self.groups[test[0]])}_out"] = self.clf.predict(self.data[test])
                self.gt[f"lopo_with_{int(self.groups[test[0]])}_out"] = self.label[test]
                self.train_size[f"lopo_with_{int(self.groups[test[0]])}_out"] = self.data[train].shape[0]
                self.test_size[f"lopo_with_{int(self.groups[test[0]])}_out"] = self.data[test].shape[0]

                i += 1
        elif 'fold' in self.cv_method:
            self.precision = {}
            self.recall = {}
            i = 1
            for train, test in self.kf.split(self.data):
                if self.standardize:
                    self.data[train] = self.scaler.fit_transform(self.data[train])
                    self.data[test] = self.scaler.transform(self.data[test])

                self.clf.fit(self.data[train], self.label[train])
                self.y_pred[f"fold{i}"] = self.clf.predict(self.data[test])
                self.gt[f"fold{i}"] = self.label[test]
                self.train_size[f"fold{i}"] = self.data[train].shape[0]
                self.test_size[f"fold{i}"] = self.data[test].shape[0]
                i += 1

    def calculate_pr_metrics(self, labels: None, pr_average='weighted', verbose=True, save=True):
        from sklearn.metrics import recall_score, precision_score
        if self.cv_method == 'fixed':
            self.recognised = len(np.where((self.y_pred == self.y_test) & (self.y_pred == 1))[0])
            self.insertions = len(np.where((self.y_pred == 1) & (self.y_test == 0))[0])
            self.relevant = len(np.where(self.y_test == 1)[0])
            try:
                self.precision = self.recognised / (self.recognised + self.insertions)
            except ZeroDivisionError:
                self.precision = 0
            try:
                self.recall = self.recognised / self.relevant
            except ZeroDivisionError:
                self.recall = 0

            if verbose:
                print('Precision: {}'.format(self.precision))
                print('Recall : {}'.format(self.recall))

        elif self.cv_method == 'lopo' or 'fold' in self.cv_method:
            precision_avg = np.zeros(len(labels))
            recall_avg = np.zeros(len(labels))
            for combo in self.y_pred.keys():
                self.precision[combo] = precision_score(self.gt[combo], self.y_pred[combo], labels=labels, average=pr_average)
                self.recall[combo] = recall_score(self.gt[combo], self.y_pred[combo], labels=labels, average=pr_average)
                precision_avg += self.precision
                recall_avg += self.recall
            precision_avg /= len(self.y_pred.keys())
            recall_avg /= len(self.y_pred.keys())
            if verbose:
                for combo in self.y_pred.keys():
                    print('Combo {}'.format(combo))
                    print('Precision: {}'.format(self.precision[combo]))
                    print('Recall : {}'.format(self.recall[combo]))
                print('Average LOPO:')
                print(f'Precision: {precision_avg}')
                print(f'Recall: {recall_avg}')
        if save:
                os.makedirs('prmetrics/', exist_ok=True)
                if pr_average is None:
                    self.precision = self.precision.tolist()
                    self.recall = self.recall.tolist()
                save_json({'Precision': self.precision,
                           'Recall': self.recall},
                          'prmetrics/overview.json')

def standardize(dataset):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    return scaler, dataset
