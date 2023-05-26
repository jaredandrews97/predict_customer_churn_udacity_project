"""
Module to hold unit tests for churn functionality

This module will be used to test
    1. import_data
    2. peform_eda
    3. encode_data
    4. perform_feature_engineering
    5. train_test_model

Author: Jared Andrews
Date: 5/26/23
"""

import os
import logging

import pytest
import sys
from constants import churn_hist_fp, cust_age_hist_fp, marital_status_hist_fp, heatmap_fp, \
    total_transaction_hist_fp, roc_curve_fp, rfc_model_fp, rf_results_fp, lr_model_fp, lr_results_fp, \
    feature_importances_fp, bank_data_fp, cat_columns, final_feats

# Setup named logger
logger = logging.getLogger('test')


def block_print():
    """
    Prevents print statements from being executed and disturbing the test info printed to console
    """
    sys.stdout = open(os.devnull, 'w')


@pytest.fixture(scope="module")
def run_eda_before_tests(model_trainer):
    """
    Fixture which runs the EDA before the parameterized test runs
    """
    block_print()
    model_trainer.perform_eda()
    yield


@pytest.fixture(scope="module")
def run_model_training_before_tests(model_trainer):
    """
    Fixture which runs the model training before the parameterized test runs
    """
    block_print()
    model_trainer.train_models()
    yield


@pytest.mark.parametrize("data_fp", [bank_data_fp, 'incorrect_fp'])
def test_import(model_trainer, data_fp):
    """
    test for reading in of data
    """
    block_print()
    try:
        # set data file path based on input
        model_trainer.data_pth = data_fp
        # read in data
        model_trainer.import_data()
        df = model_trainer.data
        # Test if output dataframe is not empty
        assert df.shape[0] > 0 and df.shape[1] > 0, "Testing import_data: The file doesn't appear to have rows and " \
                                                    "columns"
        logger.info("Testing import_data: SUCCESS")

    except FileNotFoundError as err:
        if data_fp != bank_data_fp:
            # If incorrect file path, correctly throw FileNotFoundError error
            logger.info(
                "Testing import_data: SUCCESS, incorrect file path correctly threw error")
            return None
        # Fail test if no data at correct path
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    except AssertionError as err:
        logger.error(err)
        raise err


@pytest.mark.parametrize("fp,hist_name",
                         [(churn_hist_fp,
                           "Churn Histogram"),
                          (cust_age_hist_fp,
                           "Customer Age Histogram"),
                             (marital_status_hist_fp,
                              "Marital Status Histogram"),
                             (total_transaction_hist_fp,
                              "Total Transaction Histogram"),
                             (heatmap_fp,
                              "Heatmap")])
def test_eda(run_eda_before_tests, fp, hist_name):
    """
    test perform eda function
    """
    try:
        # Check that file correctly saved
        assert os.path.isfile(
            fp), f"ERROR: test_eda did not produce {hist_name} file"
        logger.info(
            "SUCCESS: test_eda successfully produced %s file",
            hist_name)
    except AssertionError as err:
        logger.error(err)
        raise err


def test_encoder_helper(model_trainer):
    """
    test encoder helper
    """
    try:
        # Run encoder
        model_trainer.encoder_helper()
        # Get output encoded data
        enc_df = model_trainer.data
        # Check that encoded dataframe is not empty
        assert enc_df is not None and enc_df.shape[0] != 0 and enc_df.shape[
            1] != 0, "ERROR: test_encoder_helper returned empty dataframe"
        # Check that encoded dataframe has all desired encoded columns
        assert all([f"{c}_Churn" in enc_df.columns for c in cat_columns]
                   ), "ERROR: test_encoder_helper not all encoded columns returned"
        logger.info(
            "SUCCESS: test_encoder_helper successfully returned encoded dataframe")
    except AssertionError as err:
        logger.error(err)
        raise err


def test_perform_feature_engineering(model_trainer):
    """
    test perform_feature_engineering
    """
    block_print()
    try:
        # Run feature engineering step
        model_trainer.perform_feature_engineering()
        # Get outputted dataframes from feature engineering step
        x_train, x_test, y_train, y_test = model_trainer.x_train, model_trainer.x_test, \
            model_trainer.y_train, model_trainer.y_test
        # Check that all dataframe are not empty
        assert all([len(df) > 0 for df in [x_train, x_test, y_train, y_test]]), "ERROR: " \
                                                                                "test_perform_feature_engineering " \
                                                                                "at least one returned dataframe empty"
        # Check that x dataframes have desired columns
        assert all(x_train.columns == final_feats) and all(x_test.columns == final_feats), \
            "ERROR: test_perform_feature_engineering did not return proper columns for dataframes"
        logger.info(
            "SUCCESS: test_perform_feature_engineering successfully returned train/test split dataframes")
    except AssertionError as err:
        logger.error(err)
        raise err


@pytest.mark.parametrize("file_path,output_name",
                         [(lr_model_fp,
                           "Logistic Regression classifier model"),
                          (rfc_model_fp,
                           "Random Forest classifier mode"),
                             (feature_importances_fp,
                              "feature importance"),
                             (lr_results_fp,
                              "Logistic Regression classifier classification report"),
                             (rf_results_fp,
                              "Random Forest classifier classification report"),
                             (roc_curve_fp,
                              "ROC Curve")])
def test_train_models(run_model_training_before_tests, file_path, output_name):
    """
    test train_models
    """
    try:
        # Check that file correctly saved
        assert os.path.isfile(
            file_path), f"ERROR: test_train_models did not produce {output_name} file"
        logger.info(
            "SUCCESS: test_eda successfully produced %s file",
            output_name)
    except AssertionError as err:
        logger.error(err)
        raise err
