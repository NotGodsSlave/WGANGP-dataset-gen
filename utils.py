import numpy as np
import pandas as pd

def normalize_and_save(dataframe : pd.DataFrame):
    mins = pd.min()
    maxs = pd.max()
    res = pd
    for column in res.columns:
        res[column] -= mins[column]
        res[column] /= (maxs[column] - mins[column])
        res[column] *= 2
        res[column] -= 1
    
    return res, mins, maxs

def denormalize(mins, maxs, dataframe: pd.DataFrame):
    res = pd
    for column in res.columns:
        res[column] += 1
        res[column] /= 2
        res[column] *= (maxs[column] - mins[column])
        res[column] += mins[column]
    return res

from matplotlib import pyplot as plt

def build_hist(df1 : pd.DataFrame, df2 : pd.DataFrame, num_x, num_y, columns, figsize=(12,12)):
    fig, axs = plt.subplots(num_x, num_y, figsize = figsize)
    for i in range(num_x):
        for j in range(num_y):
            axs[i][j].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
            #axs[i][j].set_xlabel("Value")
            #axs[i][j].set_ylabel("# of Samples")
            axs[i][j].set_title(columns[i*num_x+j])
            minval = min(df1.min()[columns[i*num_x+j]],df2.min()[columns[i*num_x+j]])
            maxval = max(df1.max()[columns[i*num_x+j]],df2.max()[columns[i*num_x+j]])
            binwidth = (maxval-minval)/20
            axs[i][j].hist(df1[columns[i*num_x+j]],color="blue",bins=np.arange(minval,maxval+binwidth,binwidth), alpha=0.5)
            axs[i][j].hist(df2[columns[i*num_x+j]],color="red",bins=np.arange(minval,maxval+binwidth,binwidth), alpha=0.5)
    plt.show()