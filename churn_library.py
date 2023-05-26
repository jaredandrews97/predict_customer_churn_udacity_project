"""
Module which stores functions for the churn project

This module implemts the following functionality:
    1. import_data
    2. peform_eda
    3. encode_data
    4. perform_feature_engineering
    5. train_models

Author: Jared Andrews
Date: 5/26/23
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
from constants import plot_dpi, churn_hist_fp, cust_age_hist_fp, marital_status_hist_fp, quant_columns, \
    heatmap_fp, total_transaction_hist_fp, cat_columns, target_feat, final_feats, roc_curve_fp, rfc_model_fp, \
    rf_results_fp, rf_param_grid, lr_model_fp, lr_results_fp, feature_importances_fp, bank_data_fp

# Disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class ModelTrainer:
    """
       A class to represent a person.

       ...

       Attributes
       ----------
       pth : str
           path for input data

       Methods
       -------
       info(additional=""):
           Prints the person's name and age.
       """

    def __init__(self, pth, is_test=False):

        # Path for data
        self.data_pth = pth
        # Data read in during import function and used in EDA and feature_engineering
        self.data = None
        # X train data produced by feature_engineering
        self.x_train = None
        # X test data produced by feature_engineering
        self.x_test = None
        # y train data produced by feature_engineering
        self.y_train = None
        # y test data produced by feature_engineering
        self.y_test = None
        # Setup named logger
        self.logger = logging.getLogger('results')
        self.logger.disabled = True if is_test else False

    def import_data(self):
        """
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                data: pandas dataframe
        """
        print('START: import_data')
        try:
            data = pd.read_csv(self.data_pth, index_col=0)
            # Check that data has proper shape (not empty)
            assert data.shape[0] > 0 and data.shape[1] > 0
            self.logger.info("SUCCESS: File read in with %s rows", data.shape[0])
            print('END: import_data\n')
            self.data = data

        except FileNotFoundError as err:
            self.logger.error("ERROR: Filepath invalid\n %s", err)
            raise err

        except AssertionError as err:
            self.logger.error("Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    @staticmethod
    def save_plot(plot, file_path, title=None, x_label=None, y_label=None):
        """
        saves plot to file
        input:
                plot: matplotlib plot
                fp: file path to save file to

        output:
                None
        """
        # get figure from plot
        if title:
            plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        fig = plot.get_figure()
        # save to file
        fig.savefig(file_path, dpi=plot_dpi)
        # clear plot for next image
        plt.clf()

    def perform_eda(self):
        """
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        """
        data = self.data
        print('START: perform_eda')
        # Check that the data read in is a pandas dataframe
        if not isinstance(data, pd.DataFrame):
            self.logger.error(
                "ERROR: perform_eda input of type %s, desired type is Pandas dataframe",
                type(data))
            raise TypeError(
                f"Input to perform_eda is of type {type(data)}, needs to be of type Pandas dataframe")

        # Print initial EDA results to console
        print(f"Shape of data: {data.shape}\n")
        print(f"Nulls by column: \n{data.isnull().sum().to_string()}\n")
        print(f"Descriptive statistics: \n{data.isnull().sum().to_string()}\n")

        # Calculate Churn target variable
        data['Churn'] = data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        # Plot churn, customer age and total transaction histograms and martial
        # status value count barplot
        plt.figure(figsize=(20, 10))
        self.save_plot(data['Churn'].hist(), churn_hist_fp, "Distribution of Customer Churn", "Churn (Indicator)",
                       "Count")
        self.save_plot(data['Customer_Age'].hist(), cust_age_hist_fp, "Distribution of Customer Age", "Customer Age",
                       "Count")
        self.save_plot(data['Marital_Status'].value_counts(
            'normalize').plot(kind='bar'), marital_status_hist_fp, "Distribution of Customer Marital Status", None,
                       "Normalized Count")
        self.save_plot(
            sns.histplot(
                data['Total_Trans_Ct'],
                stat='density',
                kde=True),
            total_transaction_hist_fp, "Distribution of Customer Transaction Count")
        self.save_plot(
            sns.heatmap(
                data[quant_columns].corr(),
                annot=False,
                cmap='Dark2_r',
                linewidths=2),
            heatmap_fp)

        self.logger.info("SUCCESS: EDA plots saved to file")
        print('END: perform_eda\n')

    def encoder_helper(self):
        """
        helper function to turn each categorical column into a new column with
        proportion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe

        output:
                df: pandas dataframe with new columns for
        """
        # Encode each categorical column
        df = self.data
        for col in cat_columns:
            # Calculate the mean Churn rate for each value in each column
            try:
                groups = df.groupby(col)[target_feat].mean()
            except KeyError as err:
                self.logger.info('%s feature not found in inputted dataframe', col)
                raise err

            # Set mean churn rate values for encoded col
            df[f'{col}_Churn'] = df[col].map(groups)
            self.logger.info('%s feature successfully encoded', col)
        self.data = df

    def perform_feature_engineering(self):
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
        self.encoder_helper()
        # Create x,y splits
        final_x, final_y = self.data[final_feats], self.data[target_feat]

        # Perform train/test split for training
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            final_x, final_y, test_size=0.3, random_state=42)
        print('END: perform_feature_engineering\n')

    def classification_report_image(self,
                                    y_train_preds,
                                    y_test_preds,
                                    model,
                                    class_report_fp):
        """
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
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
                    self.y_train, y_train_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        self.logger.info(
            "SUCCESS: Train classification report successfully for %s",
            model)

        # Plot test classification report for model
        plt.text(0.01, 0.6, str(f'{model} Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    self.y_test, y_test_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        self.logger.info(
            "SUCCESS: Test classification report successfully for %s",
            model)

        plt.axis('off')
        plt.savefig(class_report_fp)

    print('END: perform_feature_engineering\n')

    def feature_importance_plot(self, model, output_pth):
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
        names = [self.x_train.columns[i] for i in indices]

        # Create plot
        plt.clf()
        plt.figure(figsize=(20, 20))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(self.x_train.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(self.x_train.shape[1]), names, rotation=45)
        plt.savefig(output_pth)
        self.logger.info("SUCCESS: Feature importance plot saved to file")

        print('END: feature_importance_plot\n')

    def save_roc_curve_plot(self, lrc_model, rfc_model):
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
        _, axis = plt.subplots()
        # Create Logistic Regression ROC Curve
        RocCurveDisplay.from_estimator(lrc_model, self.x_test, self.y_test, ax=axis)
        self.logger.info("SUCCESS: Logistic Regression ROC Curve successfully created")
        # Create Random Forest ROC Curve
        RocCurveDisplay.from_estimator(rfc_model, self.x_test, self.y_test, ax=axis)
        self.logger.info("SUCCESS: Random Forest ROC Curve successfully created")
        axis.figure.savefig(roc_curve_fp, dpi=300)
        plt.clf()
        print('END: save_roc_curve_plot\n')

    @staticmethod
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

    def train_models(self):
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
        cv_rfc.fit(self.x_train, self.y_train)
        self.logger.info("SUCCESS: HP Tuning successfully completed for RFC")

        # Fit LRC model
        print('Performing Training for LRC')
        lrc.fit(self.x_train, self.y_train)
        self.logger.info("SUCCESS: Training successfully completed for LRC")

        print('Getting Train/Test Predictions for LRC and RFC')
        # Get predictions for random Forest Classifier
        y_train_preds_rf = cv_rfc.best_estimator_.predict(self.x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(self.x_test)

        # Get predictions for LR Classifier
        y_train_preds_lr = lrc.predict(self.x_train)
        y_test_preds_lr = lrc.predict(self.x_test)

        print('Producing Classification report images for LRC and RFC')
        # Generate and save classification report images for RFC
        self.classification_report_image(y_train_preds_rf,
                                         y_test_preds_rf,
                                         'Random Forest',
                                         rf_results_fp)

        # Generate and save classification report images for LRC
        self.classification_report_image(y_train_preds_lr,
                                         y_test_preds_lr,
                                         'Logistic Regression',
                                         lr_results_fp)

        print('Generate and save feature importance images for RFC')
        # Generate and save feature importance images for RFC
        self.feature_importance_plot(cv_rfc, feature_importances_fp)

        print('Generate and save ROC Curve images for RFC and LRC')
        # Generate and save ROC Curve images for RFC and LRC
        self.save_roc_curve_plot(lrc, cv_rfc.best_estimator_)

        print('Saving best models')
        # Save best models
        self.save_model(cv_rfc.best_estimator_, rfc_model_fp)
        self.save_model(lrc, lr_model_fp)
        self.logger.info("SUCCESS: Model training and testing complete")
        print('END: train_models\n')


if __name__ == "__main__":

    # Setup logging configuration
    logging.basicConfig(
        filename='./logs/results.log',
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')

    model_trainer = ModelTrainer(bank_data_fp)
    model_trainer.import_data()
    model_trainer.perform_eda()
    model_trainer.perform_feature_engineering()
    model_trainer.train_models()
    print("--Done--")
