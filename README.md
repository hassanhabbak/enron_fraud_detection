# enron_fraud_detection
Fraud detection ML on Enron data


# Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

# Summary
The goal of the project is to be able to predict if an employee was deemed a person of interest by the court during the proceedings of Enron case. A person of interest is defined as someone who was either charged, settled or testified in court. Machine learning is a very well suited for this as it falls under a classification problem. Using classification algorithms, we can try to predict if an employee was POI or not. In the dataset, we have both email information and financial information.

For the Email information it contains how many emails where sent and received as well as how many were sent to or received from a POI. This is very useful if we are to try and see if there is a connection between that employee and POI individuals.

For financial data, we have the salary and how much they received as bonus. This should be a good indication on what they did for Enron as well as if they were involved in fraud in one way or another.

# Dataset information

We have 146 records in our dataset. 

There are 17 person of interest in our dataset.

There is a large number of null values in them.
    
    Null Values:
    salary                        51
    deferral_payments            107
    total_payments                21
    loan_advances                142
    bonus                         64
    restricted_stock_deferred    128
    deferred_income               97
    total_stock_value             20
    expenses                      51
    exercised_stock_options       44
    other                         53
    long_term_incentive           80
    restricted_stock              36
    director_fees                129
    
Due to the large null values, I opted to set the values to 0. Originally I tried setting the values to the mean of the other values, however due to our small dataset, it caused information dilution and the models did not train as well.

For Emails there was some really high outliers up to 15100 emails. This could be due to an error in the data. I opted to remove any outliers higher than 3 * STD value.

For salaries, while there were outliers, I opted not to remove them. That is because I believe they could be an indication to a POI.

# Features used
    . deferral_payments score: 0.421996
    . total_payments score: 0.339702
    . loan_advances score: 2.637120
    . deferred_income score: 0.076958
    . total_stock_value score: 0.188827
    . exercised_stock_options score: 0.251093
    . director_fees score: 0.531313
    . to_messages score: 1.992600
    . from_poi_to_this_person score: 3.143320
    . from_this_person_to_poi score: 3.181665
    . shared_receipt_with_poi score: 6.255575
    . bonus_per_salary score: 5.753669
    . ratio_emails_from_poi score: 3.997957
    . ratio_emails_to_poi score: 3.999569

The last 3 features were engineered. Feature 'bonus_per_salary' is how big of a bonus was received in comparison to the salary, to highlight extremely high bonuses to salary that may highlight fraud. For both features 'ratio_emails_from_poi' and 'ratio_emails_to_poi' are crucial in highlighting how much of the person's activity was emailing to and from POI. This could highlight if the person is also a POI.

I used selectkbest for selecting the best features. Then a search grid found out that 14 features are needed for the best F1 score.

# Algorithm used

In the end I have selected Adaboost Decision tree. This resulted in a boost in precision and recall over other models, although there were certain models that were performing very close. I have tried (Bayes, Adaboost Bayes, PCA with SVM). When using PCA with Bayes it resulted in much less accuracy. However the rest were very close in performance. The reason I selected Randomforest is that it does not require any feature scaling and also benefits a lot from boosting.

# Model tuning

A lot of ML models start with early assumptions that can greatly affect the performance of the model. For that, trial and error is needed to select the correct parameters for our problem. I have used GridSearchCV to try the different combination of parameters on the model to select the correct ones. That resulted in higher precision and recall rates.

Parameters used:

    dict(feature_selection__k=[8, 10, 12, 14, 16],
         clf__n_estimators=[25,50],
         clf__algorithm=('SAMME', 'SAMME.R'),
         clf__learning_rate=[0.6,0.8,1],
         clf__base_estimator__criterion=["gini", "entropy"],
         clf__base_estimator__min_samples_leaf=[1, 2, 3, 4, 5],
         clf__base_estimator__max_depth=range(1, 5),
         clf__base_estimator__class_weight=['balanced']
         )

Those are the parameters that Adaboost can accept. After running parameters randomization the following proved to be the best parameters:

    {'clf__base_estimator__class_weight':'balanced',
     'clf__algorithm': 'SAMME',
     'clf__n_estimators': 50,
     'clf__learning_rate': 0.8,
     'clf__base_estimator__max_depth': 3,
     'clf__base_estimator__criterion': 'gini',
     'feature_selection__k': 14,
     'clf__base_estimator__min_samples_leaf': 3}
     
Without model tuning, you risk running a suboptimal model. The model will not reach its top performance. However if too much tuning is done, then the model could over fit to the data and not perform as well in the real world.

# Model validation

Model validation is to see how well your model predicts an in-depended dataset correctly. This is why in the project validation was done by spliting the dataset into training to testing data using StratifiedShuffleSplit which maintains class balance and works well with small datasets. Then using cross validator, We only training with the training data, then we predict using the test data and see how well the model performed. This is to avoid over fitting. One classic mistake is to try and predict using the testing data, which might result in an extremely high accuracy but fail to perform well in real life.

# Performance metrics

Accuracy: 0.84533       
Precision: 0.40868      
Recall: 0.35800 
F1: 0.38166     
F2: 0.36710

Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, while recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. Both precision and recall are therefore based on an understanding and measure of relevance.

We only have 17 POI vs 129 non-POI. Because of this, our data is very unbalanced toward non-POI. So relying solely on accuracy is very misleading. A model that only predicts that everyone is a non-POI will have 0.88 accuracy, but zero predictive power. In this case, precision and recall are much more accurate metrics for validating the model.