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
print('**********************')
print('***Data description***')
print('**********************')
# print("POI Count: ", labels.count(1))
df = pd.DataFrame.from_dict(data_dict, orient='index')
print('Number of rows in data: ',df.shape)
df = df.replace('NaN', np.nan)  # Replacing the string NaN to np NaN
print(df.describe())
print("POI Count: ", df['poi'].value_counts())

### Task 2: Remove outliers
with pd.option_context('display.max_rows', None, 'display.max_columns', 13):
    print(df)
print('**********************')
print('***Nan and outliers***')
print('**********************')
# Fill missing values
# Filling with median will avoid the influence of outliers
"""
df[financial_features] = df[financial_features].apply(lambda x: x.fillna(x.median()),
                                                      axis=0)  # Fill with median of each column
df[email_features] = df[email_features].fillna(df[email_features].median())  # Fill with median of each column
"""

print('Nan count')
print(len(df[financial_features]) - df[financial_features].count())

df[financial_features] = df[financial_features].fillna(0)
df[email_features] = df[email_features].fillna(df[email_features].median())  # Fill with median of each column

# Remove Row Total!
# This row is the total of all other values
# df.drop('TOTAL', inplace=True)

# Removed agency
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace=True)

print('Number of rows in data: ',df.shape)

# From the description of the dataframe, outliers are mostly in email sent
# We want to retain outliers in salary or financial
print('Email to messages outlier removed:')
print(df[np.abs(df.to_messages - df.to_messages.mean()) >= 3 * df.to_messages.std() ][['poi', 'to_messages']])
print('Email from messages outlier removed:')
print(df[np.abs(df.from_messages - df.from_messages.mean()) >= (3 * df.from_messages.std())][['poi', 'from_messages']])

df = df[(np.abs(df.to_messages - df.to_messages.mean()) <= (3 * df.to_messages.std())) | (df.poi == True)]
df = df[(np.abs(df.from_messages - df.from_messages.mean()) <= (3 * df.from_messages.std())) | (df.poi == True)]
print('New Number of rows in data: ',df.shape)
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

my_dataset = df.to_dict('index')
print("Length of converted Dataset:", len(my_dataset))

### Extract features and labels from dataset for local testing
print('*************************')
print('***Extracting features***')
print('*************************')
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
print("Data row count: ", len(data))

skbest=SelectKBest(k=14)
sk_transform = skbest.fit_transform(features, labels)
indices = skbest.get_support(True)
print (skbest.scores_)

selected_features = ['poi']

for index in indices:
    print ('features: %s score: %f' % (features_list[index + 1], skbest.scores_[index]))
    selected_features.append(features_list[index + 1])

features_list = selected_features
print(features_list)

print("Final optimized features---")

labels = np.array(labels)
features = np.array(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

sss = StratifiedShuffleSplit(n_splits = 10, random_state = 42)

def print_score(clf,features, labels, sss):
    scores = cross_val_score(clf, features, labels, cv=sss, scoring='f1')
    print("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print()

def test_models():
    print('*****************************')
    print('***Trying different models***')
    print('*****************************')
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
    print('**********************')
    print('***Optimizing model***')
    print('**********************')
    print("---optmizing model---")

    sss = StratifiedShuffleSplit(500, random_state=42)

    parameters = dict(feature_selection__k=[8, 10, 12, 14, 16],
                      clf__n_estimators=[25,50],
                      clf__algorithm=('SAMME', 'SAMME.R'),
                      clf__learning_rate=[0.6,0.8,1],
                      clf__base_estimator__criterion=["gini", "entropy"],
                      clf__base_estimator__min_samples_leaf=[1, 2, 3, 4, 5],
                      clf__base_estimator__max_depth=range(1, 5),
                      clf__base_estimator__class_weight=['balanced']
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




# Running search grid on model

#clf_search = optimize_model()
#print ('Best parameters: ', clf_search.best_params_)
#clf = clf_search.best_estimator_


best_params =  {'clf__base_estimator__class_weight':'balanced',
                'clf__algorithm': 'SAMME',
                'clf__n_estimators': 50,
                'clf__learning_rate': 0.8,
                'clf__base_estimator__max_depth': 3,
                'clf__base_estimator__criterion': 'gini',
                'feature_selection__k': 14,
                'clf__base_estimator__min_samples_leaf': 3}

feature_selection = SelectKBest()
scaler = MinMaxScaler()
clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=42))

clf = Pipeline([('scaler', scaler),
                ('feature_selection', feature_selection),
                ('clf', clf)])
clf.set_params(**best_params)

print('*****************************')
print('***Testing optimized model***')
print('*****************************')
test_classifier(clf, my_dataset, features_list)


# test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print('*******************')
print('***Dumping model***')
print('*******************')
dump_classifier_and_data(clf, my_dataset, features_list)
