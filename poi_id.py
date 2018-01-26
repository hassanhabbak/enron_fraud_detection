from __future__ import print_function
# !/usr/bin/python

import sys
import pickle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

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
"""
df[financial_features] = df[financial_features].apply(lambda x: x.fillna(x.median()),
                                                      axis=0)  # Fill with median of each column
df[email_features] = df[email_features].fillna(df[email_features].median())  # Fill with median of each column
"""

df[financial_features] = df[financial_features].fillna(0)
df[email_features] = df[email_features].fillna(df[email_features].median())  # Fill with median of each column


# From the description of the dataframe, outliers are mostly in email sent
# We want to retain outliers in salary or financial
df = df[np.abs(df.to_messages - df.to_messages.mean()) <= (3 * df.to_messages.std())]
df = df[np.abs(df.from_messages - df.from_messages.mean()) <= (3 * df.from_messages.std())]

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
df['bonus_per_salary'] = df.bonus / df.salary
df['bonus_per_salary'] = df['bonus_per_salary'].fillna(0)
df['ratio_emails_from_poi'] = df.from_poi_to_this_person / df.from_messages
df['ratio_emails_to_poi'] = df.from_this_person_to_poi / df.to_messages

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock',
                 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi', 'bonus_per_salary',
                 'ratio_emails_from_poi', 'ratio_emails_to_poi']
"""
features_list = ['poi', 'salary', 'total_stock_value', 'expenses', 'bonus',
          'exercised_stock_options', 'deferred_income',
          'ratio_emails_to_poi', 'from_poi_to_this_person', 'ratio_emails_from_poi',
          'shared_receipt_with_poi']"""
print(features_list)

my_dataset = df.to_dict('index')
# my_dataset = data_dict

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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support, make_scorer, \
    f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

sss = StratifiedShuffleSplit(n_splits = 10, random_state = 42)

def print_score(clf,features, labels, sss):
    scores = cross_val_score(clf, features, labels, cv=sss, scoring='f1')
    print("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()


# Provided to give you a starting point. Try a variety of classifiers.

def test_models():
    print()
    print('------Model Accuracies')

    # Naive Bayes
    from sklearn.naive_bayes import GaussianNB

    clf = Pipeline([('reduce_dim', PCA()), ('clf', GaussianNB())])
    # clf = GaussianNB()
    print("Bayes model")
    print_score(clf,features, labels, sss)

    # SVM
    from sklearn.svm import SVC, LinearSVC

    clf = Pipeline([('reduce_dim', PCA()), ('clf', SVC())])
    # clf = SVC()
    print("PCA SVM")
    print_score(clf,features, labels, sss)

    # Random Forest
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(random_state=42)
    print("Random Forest")
    print_score(clf,features, labels, sss)

    from sklearn.ensemble import AdaBoostClassifier

    # Decision tree Ada
    from sklearn.tree import DecisionTreeClassifier

    clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=42))
    print("Ada boost decision tree")
    print_score(clf,features, labels, sss)

    # Baysian Ada
    clf = AdaBoostClassifier(GaussianNB())
    print("Ada boost Baysian")
    print_score(clf,features, labels, sss)

    # Baysian PCA Ada
    from sklearn.ensemble import AdaBoostClassifier

    clf = Pipeline([('reduce_dim', PCA()), ('clf', AdaBoostClassifier(GaussianNB()))])
    print("Ada boost PCA Baysian")
    print_score(clf,features, labels, sss)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import GridSearchCV

def optimize_model():

    print("---optmizing model---")

    sss = StratifiedShuffleSplit(100, random_state=42)

    parameters = dict(feature_selection__k=[8, 10, 12, 14, 16],
                      clf__n_estimators=[25,50],
                      clf__algorithm=('SAMME', 'SAMME.R'),
                      clf__learning_rate=[0.6,0.8,1],
                      clf__base_estimator__criterion=["gini", "entropy"],
                      #clf__base_estimator__splitter=["best", "random"],
                      clf__base_estimator__min_samples_leaf=[1, 2, 3, 4, 5],
                      clf__base_estimator__max_depth=range(1, 5),
                      #clf__base_estimator__class_weight=['balanced']
                    )

    feature_selection = SelectKBest()
    scaler = MinMaxScaler()
    clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=42))

    pipe = Pipeline([('scaler', scaler),
                     ('feature_selection', feature_selection),
                     ('clf', clf)])

    clf_search = GridSearchCV(pipe
                              , parameters,
                              scoring='f1', n_jobs=-1,
                              cv=sss, verbose=5, return_train_score=True)
    clf_search.fit(features, labels)

    return clf_search





# clf_search.fit(features, labels)

# print(clf_search.best_score_)
# print(clf_search.best_params_)

# Best parameters Precision of 1

# best_params = {'clf__algorithm': 'SAMME.R', 'clf__base_estimator__splitter': 'random', 'clf__n_estimators': 5, 'clf__learning_rate': 0.5, 'clf__base_estimator__min_samples_split': 4, 'clf__base_estimator__criterion': 'gini', 'clf__base_estimator__min_samples_leaf': 4}
# best_params = clf_search.best_params_
#clf = Pipeline([('feature_selection', SelectKBest(k=10)),
#                ('clf', AdaBoostClassifier(DecisionTreeClassifier(random_state=42)))])
#clf.set_params(**best_params)
clf_search = optimize_model()
print ('Best parameters: ', clf_search.best_params_)
clf = clf_search.best_estimator_
print("Optimized model")
test_classifier(clf, my_dataset, features_list)



# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, class_weight='balanced'),
#                          n_estimators=50, learning_rate=.8)

# test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
