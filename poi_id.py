from __future__ import print_function
# !/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

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

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.

print()
print('------Model Accuracies')

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

clf = Pipeline([('reduce_dim', PCA()), ('clf', GaussianNB())])
#clf = GaussianNB()
print("Bayes: ", accuracy_score(clf.fit(features_train, labels_train).predict(features_test),labels_test))

# SVM
from sklearn.svm import SVC
clf = Pipeline([('reduce_dim', PCA()), ('clf', SVC())])
#clf = SVC()
print("PCA SVM: ", accuracy_score(clf.fit(features_train, labels_train).predict(features_test),labels_test))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=42)
print("PCA Random Forest: ", accuracy_score(clf.fit(features_train, labels_train).predict(features_test),labels_test))

# Random forest Ada
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(RandomForestClassifier(random_state=42))
print("Ada boost Random Forest: ", accuracy_score(clf.fit(features_train, labels_train).predict(features_test),labels_test))

# Baysian Ada
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(GaussianNB())
print("Ada boost Baysian: ", accuracy_score(clf.fit(features_train, labels_train).predict(features_test),labels_test))

# Baysian PCA Ada
from sklearn.ensemble import AdaBoostClassifier
clf = Pipeline([('reduce_dim', PCA()), ('clf', AdaBoostClassifier(GaussianNB()))])
print("Ada boost PCA Baysian: ", accuracy_score(clf.fit(features_train, labels_train).predict(features_test),labels_test))


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

"""
parameters = {'n_estimators': range(10, 50), 'learning_rate':np.arange(0.1, 1, 0.1), 'algorithm':('SAMME', 'SAMME.R')}
clf_search = GridSearchCV(AdaBoostClassifier(RandomForestClassifier(random_state=42)), parameters,
                          scoring=['f1', 'precision', 'recall'], refit='precision',
                          n_jobs=4, cv=StratifiedKFold(labels))

clf_search.fit(features, labels)

print (clf_search.best_score_)
print (clf_search.best_params_)
"""
# Best parameters Precision of 1
best_params = {'n_estimators':22, 'learning_rate':0.4, 'algorithm':'SAMME.R'}
clf = AdaBoostClassifier(RandomForestClassifier(random_state=42))
clf.set_params(**best_params)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
