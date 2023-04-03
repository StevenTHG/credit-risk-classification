# Credit Risk Classification

## Overview of the Analysis

* The purpose of this analysis is to use the Logistic Regression machine learning model to predict the creditworthiness (healthy loan versus high-risk loans) of borrowers given lending activity data and assess its performance. 

* This dataset contained in the "Resources" folder has columns labelled `loan_size`, `interest_rate`, `borrower_income`, `debt_to_income`, `number_of_accounts`, `derotatory_marks`, `total_debt`, and `loan_status`. This final column has values `0` for healthy loans and `1` for high-risk loans. Here, we have 77,536 total values split between 75,036 healthy loans and 2,500 high-risk loans. 

* The first task was to split the data into training and testing sets. To do so, we start by reading the give CSV file into a DataFrame, then separate the DataFrame into features and labels (outcome) to be split by SKLearn's `train_test_split` function creating training features, training labels, testing features, and testing labels. 

* The next task was to create the Logistic Regression model. This was imported from SKLearn and used to fit the training features and labels, which allows the model to predict the testing labels using the testing features. Here, we assessed the model's performance by calculating its balance accuracy score, confusion matrix, and classification report. 

* Finally, for the sake of comparison, we imported and used Imblearn's `RandomOverSampler` function on our original data to increase the size of the high-risk loan data. Using this new dataset, we re-ran the Logistic Regression model and calculated the necessary elements to assess its performance. 

## Results

* Machine Learning Model 1: 
    * Balanced accuracy score: 0.94
    * Accuracy score: 0.99
    * Precision:
        * `0`: 1.00
        * `1`: 0.87
    * Recall:
        * `0`: 1.00
        * `1`: 0.89

* Machine Learning Model 2: 
    * Balanced accuracy score: 1.00
    * Accuracy score: 1.00
    * Precision:
        * `0`: 1.00
        * `1`: 0.87
    * Recall:
        * `0`: 1.00
        * `1`: 1.00

## Summary

Both models found overwhelming success when predicting healthy loans with precision and recall values of 1.00. In terms of prediciting high-risk loans, we can see a clear improvement in the second model in comparison to the first. While the second model showed an increase in recall, meaning the the model had a lower false negative rate, its precision stayed the same. As a consequence, the second model did not improve its rate of detecting false positives. In practice, this means that the model is more likely to predict a loan to be high-risk when it may not than to predict that the loan is healthy when it may not. When looking at this in conjuction with the low amount of high-risk loans from the original dataset, this model could be recommended in the real world on the basis such loans that are classified as high-risk be passed on to be further evaluated to confirm its validity. However, for the sake of efficiency, the model could be trained on more data.   