# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This script can be used to run the EDA and model training for the prediction of user churn based on the data we have gathered for various customers
## Files and data description
`./data/bank_data.csv` : Raw data that is used to train the model. Contains demographic and financial information on the customers
`./images/eda/` : Contains images of results obtained after running the EDA script
`./images/results/` : Contains images of classification report for the models created
`./logs/` : Contains logs for the script
`./models/`  : Contains models that are generated
## Running Files

The program can be run by running the churn_library.py file from the terminal. You can use the following command to execute it
`python churn_library.py`

To test the function, you can use the `churn_script_logging_and_tests.py` file, with the pytest library
`pytest churn_script_logging_and_tests.py`

The logs for the script will be recorded in `./logs/churn_library.log`

The results of the script are stored in the `./images` and `./models` folders 
