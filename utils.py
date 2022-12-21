import pandas as pd
import os
from google.colab import drive

drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/Colab Notebooks/daicon/logistics_competition/')


def get_larger_range_first_last_utils(v1, df, num=3):
    new_df = df.copy()
    num_lst = []
    for i in range(len(df)):
        str_num_first = str(df.iloc[i, 1])[:num]
        str_num_last = str(df.iloc[i, 1])[-num:]
        # print(str_num)
        num_lst.append(int(str_num_first + str_num_last))
    new_df['{}_FIRST_LAST_{}'.format(v1, num)] = num_lst
    return new_df


def get_larger_range_first_utils(v1, df, num):
    new_df = df.copy()
    num_lst = []
    for i in range(len(new_df)):
        str_num = str(new_df.iloc[i, 1])[:num]
        # print(str_num)
        num_lst.append(int(str_num))
    new_df['{}_FIRST_{}'.format(v1, num)] = num_lst
    return new_df


def get_larger_range_last_utils(v1, df, num):
    new_df = df.copy()
    num_lst = []
    for i in range(len(new_df)):
        str_num = str(new_df.iloc[i, 1])[-num:]
        # print(str_num)
        num_lst.append(int(str_num))
    new_df['{}_LAST_{}'.format(v1, num)] = num_lst
    return new_df
