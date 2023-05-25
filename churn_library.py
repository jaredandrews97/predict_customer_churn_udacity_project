"""
Module which stores functions for the churn project
"""

import os
import pickle
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from constants import plot_dpi, churn_hist_fp, cust_age_hist_fp, marital_status_hist_fp, quant_columns,\
    heatmap_fp, total_transaction_hist_fp, cat_columns, target_feat, final_feats, roc_curve_fp, rfc_model_fp, \
    rf_results_fp, rf_param_grid, lr_model_fp, lr_results_fp, feature_importances_fp, bank_data_fp

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Setup logging configuration
logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    """
    print('START: import_data')
    try:
        data = pd.read_csv(pth, index_col=0)
        assert data.shape[0] > 0 and data.shape[1] > 0
        logging.info("SUCCESS: File read in with %s rows", data.shape[0])
        print('END: import_data\n')
        return data

    except FileNotFoundError as err:
        logging.error("ERROR: Filepath invalid\n %s", err)
        raise err

    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def save_plot(plot, fp):
    """
    saves plot to file
    input:
            plot: matplotlib plot
            fp: file path to save file to

    output:
            None
    """
    fig = plot.get_figure()
    fig.savefig(fp, dpi=plot_dpi)
    plt.clf()


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """

    print('START: perform_eda')
    if not isinstance(df, pd.DataFrame):
        logging.error(
            "ERROR: perform_eda input of type %s, desired type is Pandas dataframe",
            type(df))
        raise TypeError(
            f"Input to perform_eda is of type {type(df)}, needs to be of type Pandas dataframe")

    # Print initial EDA results to console
    print(f"Shape of data: {df.shape}\n")
    print(f"Nulls by column: \n{df.isnull().sum().to_string()}\n")
    print(f"Descriptive statistics: \n{df.isnull().sum().to_string()}\n")

    # Calculate Churn target variable
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Plot churn, customer age and total transaction histograms and martial
    # status value count barplot
    plt.figure(figsize=(20, 10))
    save_plot(df['Churn'].hist(), churn_hist_fp)
    save_plot(df['Customer_Age'].hist(), cust_age_hist_fp)
    save_plot(df['Marital_Status'].value_counts(
        'normalize').plot(kind='bar'), marital_status_hist_fp)
    save_plot(
        sns.histplot(
            df['Total_Trans_Ct'],
            stat='density',
            kde=True),
        total_transaction_hist_fp)
    save_plot(
        sns.heatmap(
            df[quant_columns].corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2),
        heatmap_fp)

    logging.info("SUCCESS: EDA plots saved to file")
    print('END: perform_eda\n')


def encoder_helper(df):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe

    output:
            df: pandas dataframe with new columns for
    """
    # Encode each categorical column
    for col in cat_columns:
        col_lst = []
        # Calculate the mean Churn rate for each value in each column
        try:
            card_groups = df.groupby(col)[target_feat].mean()
        except KeyError as err:
            logging.info('%s feature not found in inputted dataframe', col)
            raise err

        for val in df[col]:
            col_lst.append(card_groups.loc[val])

        # Set mean churn rate values for encoded col
        df[f'{col}_Churn'] = col_lst
        logging.info('%s feature successfully encoded', col)
    return df


def perform_feature_engineering(df):
    """
    input:
              df: pandas dataframe

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    print('START: perform_feature_engineering')
    # Encode the categorical features
    encoded_df = encoder_helper(df)
    # Create x,y splits
    final_x, final_y = encoded_df[final_feats], encoded_df[target_feat]

    # Perform train/test split for training
    x_train, x_test, y_train, y_test = train_test_split(
        final_x, final_y, test_size=0.3, random_state=42)
    print('END: perform_feature_engineering\n')
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                model,
                                class_report_fp):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    print('START: perform_feature_engineering')
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    # Plot train classification report for model
    plt.text(0.01, 1.25, str(f'{model} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    logging.info(
        "SUCCESS: Train classification report successfully for %s",
        model)

    # Plot test classification report for model
    plt.text(0.01, 0.6, str(f'{model} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    logging.info(
        "SUCCESS: Test classification report successfully for %s",
        model)

    plt.axis('off')
    plt.savefig(class_report_fp)

    print('END: perform_feature_engineering\n')


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    print('START: feature_importance_plot')

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.clf()
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    logging.info("SUCCESS: Feature importance plot saved to file")

    print('END: feature_importance_plot\n')


def save_roc_curve_plot(lrc_model, rfc_model, xtest, ytest):
    """
    Save roc curve plot for RFC and LRC
    input:
            lrc_model: fitted logistic regression classifier model
            rfc_model: fitted random forest classifier model
            xtest: input features for test set
            ytest: output feature for test set
    """
    print('START: save_roc_curve_plot\n')
    plt.clf()
    plt.figure(figsize=(15, 8))
    _, ax = plt.subplots()
    # Create Logistic Regression ROC Curve
    RocCurveDisplay.from_estimator(lrc_model, xtest, ytest, ax=ax)
    logging.info("SUCCESS: Logistic Regression ROC Curve successfully created")
    # Create Random Forest ROC Curve
    RocCurveDisplay.from_estimator(rfc_model, xtest, ytest, ax=ax)
    logging.info("SUCCESS: Random Forest ROC Curve successfully created")
    ax.figure.savefig(roc_curve_fp, dpi=300)
    plt.clf()
    print('END: save_roc_curve_plot\n')


def save_model(model, model_fp):
    """
    Helper function to save trained model files
    input:
            model: fit model to be saved
            model_fp: file path where model is saved to
    output:
            None
    """
    with open(model_fp, 'wb') as file_path:
        pickle.dump(model, file_path)


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    print('START: train_models')
    # Instantiate RFC and LRC models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Perform HP tuning using CV for Random Forest Classifier; use best model
    print('Performing HP Tuning for RFC')
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=rf_param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    logging.info("SUCCESS: HP Tuning successfully completed for RFC")

    # Fit LRC model
    print('Performing Training for LRC')
    lrc.fit(x_train, y_train)
    logging.info("SUCCESS: Training successfully completed for LRC")

    print('Getting Train/Test Predictions for LRC and RFC')
    # Get predictions for random Forest Classifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    # Get predictions for LR Classifier
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    print('Producing Classification report images for LRC and RFC')
    # Generate and save classification report images for RFC
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_rf,
                                y_test_preds_rf,
                                'Random Forest',
                                rf_results_fp)

    # Generate and save classification report images for LRC
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_test_preds_lr,
                                'Logistic Regression',
                                lr_results_fp)

    print('Generate and save feature importance images for RFC')
    # Generate and save feature importance images for RFC
    feature_importance_plot(cv_rfc, x_train, feature_importances_fp)

    print('Generate and save ROC Curve images for RFC and LRC')
    # Generate and save ROC Curve images for RFC and LRC
    save_roc_curve_plot(lrc, cv_rfc.best_estimator_, x_test, y_test)

    print('Saving best models')
    # Save best models
    save_model(cv_rfc.best_estimator_, rfc_model_fp)
    save_model(lrc, lr_model_fp)
    logging.info("SUCCESS: Model training and testing complete")
    print('END: train_models\n')


if __name__ == "__main__":
    bank_df = import_data(bank_data_fp)
    perform_eda(bank_df)
    train_x, test_x, train_y, test_y = perform_feature_engineering(bank_df)
    train_models(train_x, test_x, train_y, test_y)
    print("--Done--")
