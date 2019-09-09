"""
Helper functions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

ZEROS = np.zeros(5)
ONES = np.ones(10)

#Takes Dataframe as input and returns NaNs for each column. Output: Pandas Series
def df_nulls(df):
    print(f'Number of null values in each column is: \n{df.isnull().sum(axis = 0)}.')

#inputs true and corresponding Predicted values and plots confusion matrix 
def plot_confusion_matrix(y_true,y_pred,normalize=False):
    columns = [f'Predicted "{c}"'for c in unique_labels(y_pred)]
    index_names = [f'Actual "{c}"'for c in unique_labels(y_true)]
    cm = confusion_matrix(y_true,y_pred)
    if normalize:
        cm=cm/cm.sum(axis=1).reshape(y_true.nunique(),1)
    df = pd.DataFrame(cm,columns = columns,index = index_names)
    sns.heatmap(df,cmap='viridis',annot=True,fmt='.2f');

def datetime_split(df,col):
    df[f'{col}'] = pd.to_datetime(df[f{'col'}],infer_datetime_format=True)
    df[f'{col}_year'] = df[f'{col}'].dt.year
    df[f'{col}_month'] = df[f'{col}'].dt.month
    df[f'{col}_day'] = df[f'{col}'].dt.day
    df = df.drop(columns = f'{col}')
    return df



