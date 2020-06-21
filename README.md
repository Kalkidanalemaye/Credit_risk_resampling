# Credit_risk_resampling

## Objective

#### To build and evaluate several machine learning models to assess credit risk, using data from LendingClub; a peer-to-peer lending services company.

## The goal of the challenge
1. Implement machine learning models.
2. Use resampling to attempt to address class imbalance.
3. Evaluate the performance of machine learning models.

## Intructions

1. Oversample the data using the RandomOverSampler and SMOTE algorithms.
2. Undersample the data using the cluster centroids algorithm.
3. Use a combination approach with the SMOTEENN algorithm.

#### Each step above requires to train a logistic regression classifier (from Scikit-learn) using the resampled data, calculate the balanced accuracy score using balanced_accuracy_score from sklearn.metrics, generate a confusion_matrix and print the classification report (classification_report_imbalanced from imblearn.metrics).

## Analysis

### Credit Risk Resampling
In the credit risk resampling process we used four models to test out the data. The model with the highest accuracy score is the Naive Random Oversampling, 0.71 and I would reccomend using model although, both Oversampling models close in values. It is important to note that when looking at the confusion matrix for both oversampling models, the Naive Random Oversampling does better than SMOTE Oversampling.

### Credit Risk Ensemble

Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier models were used to to test the predictions of low and high risk credit applications. Easy Ensemble AdaBoost Classifier model is the best model out of the two because the Accuracy Score is 0.93, the F1 score for low risk is 0.97, and the Recall score for high and low risk are 0.92 and 0.94, respectively. This model is able to predict credit risk applications accurately when compared to the Random Fores Classifier, therefore I would reccomend the use of Easy Ensemble AdaBoost Classifier model.
