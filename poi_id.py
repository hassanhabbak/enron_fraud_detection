from __future__ import print_function
# !/usr/bin/python

import sys
import pickle

from sklearn.feature_selection import SelectFromModel, SelectKBest

sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary']  # You will need to use more features

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                      'director_fees']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Create dataframes
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)  # Replacing the string NaN to np NaN
print(df.describe())

### Task 2: Remove outliers

# Fill missing values
# Filling with median will avoid the influence of outliers
df[financial_features] = df[financial_features].apply(lambda x: x.fillna(x.median()),
                                                      axis=0)  # Fill with median of each column
df[email_features] = df[email_features].fillna(df[email_features].median())  # Fill with median of each column

# print(df.loc['BADUM JAMES P'])

# From the description of the dataframe, outliers are mostly in email sent
# We want to retain outliers in salary or financial
df = df[np.abs(df.to_messages - df.to_messages.mean()) <= (3 * df.to_messages.std())]
df = df[np.abs(df.from_messages - df.from_messages.mean()) <= (3 * df.from_messages.std())]

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
df['bonus_per_salary'] = df.bonus / df.salary
df['ratio_emails_from_poi'] = df.from_poi_to_this_person / df.from_messages
df['ratio_emails_to_poi'] = df.from_this_person_to_poi / df.to_messages

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi', 'bonus_per_salary',
                 'ratio_emails_from_poi', 'ratio_emails_to_poi']

my_dataset = df.to_dict('index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

labels = np.array(labels)
features = np.array(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support, make_scorer, \
    f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

sss = StratifiedShuffleSplit(n_splits=1)

for train_index, test_index in sss.split(features, labels):
    print("TRAIN:", train_index, "TEST:", test_index)
    features_train, features_test = features[train_index], features[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]

# Provided to give you a starting point. Try a variety of classifiers.

print()
print('------Model Accuracies')

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

clf = Pipeline([('reduce_dim', PCA()), ('clf', GaussianNB())])
# clf = GaussianNB()
print("Bayes model")
scores = precision_recall_fscore_support(labels_test, clf.fit(features_train, labels_train).predict(features_test),
                                         average='macro')
print("Precision: ", scores[0])
print("Recall: ", scores[1])
print()

# SVM
from sklearn.svm import SVC, LinearSVC

clf = Pipeline([('reduce_dim', PCA()), ('clf', SVC())])
# clf = SVC()
print("PCA SVM")
scores = precision_recall_fscore_support(labels_test, clf.fit(features_train, labels_train).predict(features_test),
                                         average='macro')
print("Precision: ", scores[0])
print("Recall: ", scores[1])
print()

# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
print("Random Forest")
scores = precision_recall_fscore_support(labels_test, clf.fit(features_train, labels_train).predict(features_test),
                                         average='macro')
print("Precision: ", scores[0])
print("Recall: ", scores[1])
print()

from sklearn.ensemble import AdaBoostClassifier

# Decision tree Ada
from sklearn.tree import DecisionTreeClassifier

clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=42))
print("Ada boost decision tree")
scores = precision_recall_fscore_support(labels_test, clf.fit(features_train, labels_train).predict(features_test),
                                         average='macro')
print("Precision: ", scores[0])
print("Recall: ", scores[1])
print()

# Baysian Ada
clf = AdaBoostClassifier(GaussianNB())
print("Ada boost Baysian")
scores = precision_recall_fscore_support(labels_test, clf.fit(features_train, labels_train).predict(features_test),
                                         average='macro')
print("Precision: ", scores[0])
print("Recall: ", scores[1])
print()

# Baysian PCA Ada
from sklearn.ensemble import AdaBoostClassifier

clf = Pipeline([('reduce_dim', PCA()), ('clf', AdaBoostClassifier(GaussianNB()))])
print("Ada boost PCA Baysian")
scores = precision_recall_fscore_support(labels_test, clf.fit(features_train, labels_train).predict(features_test),
                                         average='macro')
print("Precision: ", scores[0])
print("Recall: ", scores[1])
print()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

"""
parameters = {"clf_max_depth": [2,3,4,5,6,7,8,9,10,11,12],
              "clf_max_features": ['auto', 'sqrt', 'log2'],
              "clf_min_samples_split": range(2, 6),
              "clf_min_samples_leaf": range(1, 11),
              "clf_n_estimators" : [5,7,10,12,14],
              "clf_criterion": ["gini", "entropy"]}
              """

parameters = dict(feature_selection__k=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                  random_forest__n_estimators=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                  random_forest__criterion=["gini", "entropy"],
                  random_forest__min_samples_split=[2, 3, 4, 5, 10])

sss = StratifiedShuffleSplit(n_splits=10, random_state=42)

pipe = Pipeline([('feature_selection', SelectKBest()),
                 ('random_forest', RandomForestClassifier())])

clf_search = GridSearchCV(pipe
                          , parameters,
                          ['precision', 'recall'], refit='recall', n_jobs=-1,
                          cv=5, verbose=5, return_train_score=True)

clf_search.fit(features, labels)

print(clf_search.best_score_)
print(clf_search.best_params_)


# Best parameters Precision of 1

best_params = {'random_forest__n_estimators': 2, 'feature_selection__k': 22, 'random_forest__min_samples_split': 10, 'random_forest__criterion': 'entropy'}
clf = Pipeline([('feature_selection', SelectKBest()),
                 ('random_forest', RandomForestClassifier())])
clf.set_params(**best_params)

print("optimized model (Adaboost Randomforest):")
prediction = clf.fit(features_train, labels_train).predict(features_test)
print(labels_test)
print()
print(prediction)
scores = precision_recall_fscore_support(labels_test, prediction, average='macro')
print("Precision: ", scores[0])
print("Recall: ", scores[1])

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
