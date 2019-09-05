"""
Helper functions
"""

import pandas as pd
import numpy as np

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

