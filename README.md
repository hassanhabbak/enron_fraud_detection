# enron_fraud_detection
Fraud detection ML on Enron data


# Project Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

# Summary
The goal of the project is to be able to predict if an employee was deemed a person of interest by the court during the proceedings of Enron case. A person of interest is defined as someone who was either charged, settled or testified in court. Machine learning is a very well suited for this as it falls under a classification problem. Using classification algorithms, we can try to predict if an employee was POI or not. In the dataset, we have both email information and financial information.

For the Email information it contains how many emails where sent and received as well as how many were sent to or received from a POI. This is very useful if we are to try and see if there is a connection between that employee and POI individuals.

For financial data, we have the salary and how much they received as bonus. This should be a good indication on what they did for Enron as well as if they were involved in fraud in one way or another.

# Features used
    .'salary'
    .'deferral_payments'
    .'total_payments'
    .'loan_advances'
    .'bonus'
    .'restricted_stock_deferred'
    .'deferred_income'
    .'total_stock_value'
    .'expenses'
    .'exercised_stock_options'
    .'other'
    .'long_term_incentive'
    .'restricted_stock'
    .'director_fees'
    .'to_messages'
    .'from_poi_to_this_person'
    .'from_messages'
    .'from_this_person_to_poi'
    .'shared_receipt_with_poi'
    .'bonus_per_salary'
    .'ratio_emails_from_poi'
    .'ratio_emails_to_poi'

The last 3 features were engineered. Feature 'bonus_per_salary' is how big of a bonus was received in comparison to the salary, to highlight extremely high bonuses to salary that may highlight fraud. For both features 'ratio_emails_from_poi' and 'ratio_emails_to_poi' are crucial in highlighting how much of the person's activity was emailing to and from POI. This could highlight if the person is also a POI.

PCA was used for dimensionality reduction in some of the models used. However the end model used is insensitive to feature scaling (Random forest), so no scaling is required.

# Algorithm used

In the end I have selected Random Forest. This resulted in a boost in precision and recall over other models, although there were certain models that were performing very close. I have tried (Bayes, Adaboost Bayes, PCA with SVM). When using PCA with Bayes it resulted in much less accuracy. However the rest were very close in performance. The reason I selected Randomforest is that it does not require any feature scaling and also deals really well with high dimensionality.

# Model tuning

A lot of ML models start with early assumptions that can greatly affect the performance of the model. For that, trial and error is needed to select the correct parameters for our problem. I have used GridSearchCV to try the different combination of parameters on the model to select the correct ones. That resulted in higher precision and recall rates.

Parameters used:

    ."max_depth": [3, None]
    ."max_features": range(1, 11)
    ."min_samples_split": range(2, 15)
    ."bootstrap": [True, False]
    ."criterion": ["gini", "entropy"]

Those are the parameters that Adaboost can accept. After running parameters randomization the following proved to be the best parameters:

    .'bootstrap':True
    .'min_samples_split':2
    .'criterion':'gini'
    .'max_features':7
    .'max_depth':None

# Model validation

Model validation is to see how well your model predicts an in-depended dataset correctly. This is why in the project validation was done by spliting the dataset into training to testing data. We only training with the training data, then we predict using the test data and see how well the model performed. This is to avoid over fitting. One classic mistake is to try and predict using the testing data, which might result in an extremely high accuracy but fail to perform well in real life.

# Performance metrics

Accuracy: 0.79
Precision: 0.404
Recall: 0.485
