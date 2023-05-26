"""
Module to store hard-coded constants used in churn_library

Author: Jared Andrews
Date: 5/26/23
"""
import os


class DataFrameColumnException(Exception):
    pass


bank_data_fp = './data/bank_data.csv'

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

quant_columns = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

plot_dpi = 300

eda_fp = './images/eda'
churn_hist_fp = os.path.join(eda_fp, 'churn_distribution.png')
cust_age_hist_fp = os.path.join(eda_fp, 'customer_age_distribution.png')
marital_status_hist_fp = os.path.join(eda_fp, 'marital_status_distribution.png')
total_transaction_hist_fp = os.path.join(eda_fp, 'total_transaction_distribution.png')
heatmap_fp = os.path.join(eda_fp, 'heatmap.png')

results_fp = './images/results'
feature_importances_fp = os.path.join(results_fp, 'feature_importances.png')
lr_results_fp = os.path.join(results_fp, 'logistic_results.png')
rf_results_fp = os.path.join(results_fp, 'rf_results.png')
roc_curve_fp = os.path.join(results_fp, 'roc_curve_result.png')

models_fp = './models'
lr_model_fp = os.path.join(models_fp, 'logistic_model.pkl')
rfc_model_fp = os.path.join(models_fp, 'rfc_model.pkl')

final_feats = ['Customer_Age', 'Dependent_count', 'Months_on_book',
               'Total_Relationship_Count', 'Months_Inactive_12_mon',
               'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
               'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
               'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
               'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
               'Income_Category_Churn', 'Card_Category_Churn']

target_feat = 'Churn'

rf_param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}

roc_alpha = 0.8

