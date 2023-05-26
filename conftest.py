import pytest
from churn_library import ModelTrainer
from constants import bank_data_fp


@pytest.fixture(scope='session', autouse=True)
def model_trainer(request):
    return ModelTrainer(bank_data_fp, is_test=True)
