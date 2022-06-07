# library doc string
'''
A library to predict customer churn
'''

# import libraries
import os
import logging
import pytest
# Other libraries
# import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report  # , plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import normalize
import seaborn as sns
sns.set()


# Variable settings
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(filename='./logs/churn_library.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %I:%M:%S %p')


def import_data(path):
    '''
    returns dataframe for the csv found at pth

    input:
            path: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_input = pd.read_csv(path)
    df_input['Churn'] = df_input['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df_input


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    logging.info("Data frame is of shape %s", df.shape)
    logging.info("Null distribution looks like the following:")
    logging.info(df.isnull().sum())
    logging.info("Description looks like the following:")
    logging.info(df.describe())

    plt.figure(figsize=(20, 10))
    df['Churn'].hist().figure.savefig('./images/eda/EDA-Churn-hist.png')
    df['Customer_Age'].hist().figure.savefig(
        './images/eda/EDA-Customer_Age-hist.png')
    df.Marital_Status.value_counts('normalize').plot(kind='bar').figure.savefig(
        './images/eda/EDA-Marital_Status-hist.png')
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True).figure.savefig(
        './images/eda/EDA-Total_Trans_Ct-density.png')
    sns.heatmap(
        df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2).figure.savefig('./images/eda/EDA-corr-heatmap.png')


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
    output:
            df: pandas dataframe with new columns for
    '''

    for category in category_lst:
        category_map = df.groupby(category).mean()['Churn'].to_dict()
        df[category + "_Churn"] = df[category].map(category_map)

    return df


def perform_feature_engineering(df, response='Churn'):
    '''
    input:
              df: pandas dataframe
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
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
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = df[keep_cols]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
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
    '''
    # RF
    fig = plt.figure()  # an empty figure with no Axes
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    fig.savefig('./images/results/Results-Classification_Report-RF.png')

    # LR
    fig = plt.figure()  # an empty figure with no Axes
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    fig.savefig('./images/results/Results-Classification_Report-LR.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
#     explainer = shap.TreeExplainer(model.best_estimator_)
#     shap_values = explainer.shap_values(X_data)
#     shap.summary_plot(shap_values, X_data, plot_type="bar")

    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    fig = plt.figure(figsize=(20, 5))  # an empty figure with no Axes

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    fig.savefig(
        os.path.join(
            output_pth,
            f'Results-Feature_Importance_Plot-{type(model).__name__}.png'))


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)


if __name__ == "__main__":

    df = import_data("./data/bank_data.csv")
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    
    df_encoded = encoder_helper(df, category_lst)

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_encoded)

    train_models(X_train, X_test, y_train, y_test)