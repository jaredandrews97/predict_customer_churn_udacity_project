"""
Module to hold unit tests for churn functionality
"""

import os
import logging

import pandas as pd
import pytest
import sys
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models
from constants import churn_hist_fp, cust_age_hist_fp, marital_status_hist_fp, heatmap_fp, total_transaction_hist_fp, \
    roc_curve_fp, rfc_model_fp, rf_results_fp, lr_model_fp, lr_results_fp, feature_importances_fp, bank_data_fp


def block_print():
    sys.stdout = open(os.devnull, 'w')


def pytest_configure():
    pytest.df = None


@pytest.fixture(scope="module", params=[bank_data_fp])
def path(request):
    value = request.param
    return value


@pytest.fixture(scope="module")
def run_eda_before_tests():
    block_print()
    perform_eda(pytest.df)
    yield


def test_import(path):
    try:
        df = import_data(path)
        pytest.df = df
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        if path != bank_data_fp:
            logging.error("Testing import_eda: Improper file wasn't found, failed as intended")
            return None
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.mark.parametrize("fp,hist_name", [(churn_hist_fp, "Churn Histogram"),
                                          (cust_age_hist_fp, "Customer Age Histogram"),
                                          (marital_status_hist_fp, "Marital Status Histogram"),
                                          (total_transaction_hist_fp, "Total Transaction Histogram"),
                                          (heatmap_fp, "Heatmap")])
def test_eda(run_eda_before_tests, fp, hist_name):
    """
    test perform eda function
    """
    try:
        assert os.path.isfile(fp), f"ERROR: test_eda did not create {hist_name} file"
        logging.info("SUCCESS: test_eda successfully produced %s file", hist_name)
    except AssertionError as err:
        logging.error(err)
        raise err


def test_encoder_helper():
    """
    test encoder helper
    """
    try:
        enc_df = encoder_helper(pytest.df)
        assert enc_df is not None and enc_df.shape[0] != 0 and enc_df.shape[1] != 0, f"ERROR: test_encoder_helper did" \
                                                                                     f" not returned proper dataframe"
        logging.info("SUCCESS: test_encoder_helper successfully returned encoded dataframe")
        pytest.df = enc_df
    except AssertionError as err:
        logging.error(err)
        raise err


def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """


def test_train_models():
    """
    test train_models
    """
