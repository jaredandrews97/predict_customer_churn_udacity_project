# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The purpose of this project is to utilize historical Udacity customer information to predict if a new customer will 
churn. The data used to train the model was provided directly from Udacity and contains 22 features for 10127 unique 
customers. Two models, a logistic regression model and a random forest classifier, were developed and analyzed, with the 
random forest classifier performing better. The analysis, data-preprocessing and training of model utilizes the 
high-level steps listed below:
1. Reading in Data
2. EDA
3. Feature Engineering
4. Model Training
5. Model Evaluation


## Files and data description
```
.
├── data
│   └── bank_data.csv                               # Input data
├── images                                      
│   ├── eda
│   │   ├── churn_distribution.png                  # Histogram of the churn target feature
│   │   ├── customer_age_distribution.png           # Histogram of the customer age input feature
│   │   ├── heatmap.png
│   │   ├── marital status_distribution.png         # Histogram of the marital status input feature
│   │   └── total_transaction_distribution.png      # Histogram with smoothed KDE curve of the total transaction count 
│   │                                                 input feature
│   └── results
│       ├── feature_importances.png                 # Barplot of feature importances for the random forest classifier
│       ├── logistic_results.png                    # Classification report for the logistic regression classifier
│       ├── rf_results.png                          # Classification report for the random forest classifier
│       └── roc_curve_results.png                   # ROC Curve of both classifiers
├── logs
│   ├── churn_library.log                           # Log produced during unit testing
│   └── results.log                                 # Log produced during model training
├── models
│   ├── logistic_model.pkl                          # Pickled file containing trained logistic regression model
│   └── rfc_model.pkl                               # Pickled file containing trained random forest model
├── churn_library.py                                # Main file containing the ModelTrainer class used to perform 
│                                                     model training
├── churn_script_logging_and_testing.py             # Unit testing functions
├── conftest.py                                     # Unit testing functions
├── constants.py                                    # Hard-coded variables 
├── pytest.ini                                      # Setting up of logger for pytest tests
├── README.md                                       # README file
└── requirements.txt                                # List of libraries/versions used to setup environment
```
## Running Files
1. Clone repo
2. Set up environment (using conda):

        conda create -n "churn_project" python=3.11.0 --file requirements.txt
3. Perform model training:

        python3 churn_library.py
4. Perform testing:
        
        pytest churn_script_logging_and_tests.py



