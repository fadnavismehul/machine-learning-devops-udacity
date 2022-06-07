'''
Script to test churn_library.py functions
'''
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data=cls.import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda=cls.perform_eda):
    '''
    test perform eda function
    '''

    df = cls.import_data("./data/bank_data.csv")
    try:
        perform_eda(df)
    except BaseException:
        logging.error("Testing perform_eda: Function is not working")

    image_file_list = [
        'EDA-Churn-hist.png',
        'EDA-Customer_Age-hist.png',
        'EDA-Marital_Status-hist.png',
        'EDA-Total_Trans_Ct-density.png',
        'EDA-corr-heatmap.png'
    ]
    try:
        for image in image_file_list:
            assert os.path.exists(os.path.join('./images/eda/', image))
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The file %s is missing", image)
        raise err


def test_encoder_helper(encoder_helper=cls.encoder_helper):
    '''
    test encoder helper
    '''
    df = cls.import_data("./data/bank_data.csv")
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    try:
        df_encoded = encoder_helper(df, category_lst)
    except Exception:
        logging.error("Testing encoder_helper: Function is not working")
        raise Exception
    # Assertion tests for result
    try:
        assert df_encoded.shape[0] > 0
        assert df_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The file doesn't appear to have rows and columns")
        raise err


def test_perform_feature_engineering(
        perform_feature_engineering=cls.perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df = cls.import_data("./data/bank_data.csv")
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_encoded = cls.encoder_helper(df, category_lst)

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df_encoded)
    except Exception:
        logging.error(
            "Testing perform_feature_engineering: Function is not working")
        raise Exception
    # Assertion checks
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The file doesn't appear to have rows and columns")
        raise err


def test_train_models(train_models=cls.train_models):
    '''
    test train_models
    '''

    df = cls.import_data("./data/bank_data.csv")
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_encoded = cls.encoder_helper(df, category_lst)

    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df_encoded)

    try:
        train_models(X_train, X_test, y_train, y_test)
    except Exception:
        logging.error(
            "Testing train_models: Function is not working")
        raise Exception

    # Checking Model Files
    try:
        assert os.path.exists('./models/rfc_model.pkl')
        assert os.path.exists('./models/logistic_model.pkl')
    except AssertionError as err:
        logging.error(
            "Testing train_models: The model files are missing")
        raise err
    # Checking Classification Report files
    try:
        assert os.path.exists(
            './images/results/Results-Classification_Report-RF.png')
        assert os.path.exists(
            './images/results/Results-Classification_Report-LR.png')
    except AssertionError as err:
        logging.error(
            "Testing train_models: The Classification reports are missing")
        raise err
