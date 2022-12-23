import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold

from utils import *


def get_data():
    # 데이터 불러오기
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    return train_df, test_df


def get_rename(train_df, test_df):
    train_df.rename(columns={'송하인_격자공간고유번호': 'SGRID'}, inplace=True)
    train_df.rename(columns={'수하인_격자공간고유번호': 'CGRID'}, inplace=True)
    train_df.rename(columns={'물품_카테고리': 'ITEM'}, inplace=True)
    train_df.rename(columns={'운송장_건수': 'TARGET'}, inplace=True)

    test_df.rename(columns={'송하인_격자공간고유번호': 'SGRID'}, inplace=True)
    test_df.rename(columns={'수하인_격자공간고유번호': 'CGRID'}, inplace=True)
    test_df.rename(columns={'물품_카테고리': 'ITEM'}, inplace=True)
    return train_df, test_df


def get_one_hot_encoding(v1, train_df, test_df):
    # 레이블 인코더, 원-핫 인코더 생성
    label_encoder_item = LabelEncoder()
    onehot_encoder = OneHotEncoder()
    v1_numpy_train = train_df[v1]
    # 레이블 인코딩 적용(문자 데이터 -> 숫자 데이터)
    label_encoder_item.fit(v1_numpy_train)
    items_label_encoded = label_encoder_item.transform(v1_numpy_train)
    # 원-핫 인코딩 적용
    items_onehot_encoded = onehot_encoder.fit_transform(items_label_encoded.reshape(-1, 1))
    enc_train = items_onehot_encoded.toarray()
    # oenc.get_feature_names() use this method to get column names
    train_v1 = pd.DataFrame(enc_train, columns=onehot_encoder.get_feature_names())

    onehot_encoder = OneHotEncoder()
    items_test = test_df[v1]
    # 레이블 인코딩 적용(문자 데이터 -> 숫자 데이터)
    items_label_encoded_test = label_encoder_item.transform(items_test)
    # 원-핫 인코딩 적용
    items_test_onehot_encoded = onehot_encoder.fit_transform(items_label_encoded_test.reshape(-1, 1))
    enc_test = items_test_onehot_encoded.toarray()
    test_v1 = pd.DataFrame(enc_test, columns=onehot_encoder.get_feature_names())
    return train_v1, test_v1


def kfold_target_encoder(train, test, cols_encode, target, folds=10):
    ## KFOLD TARGET 인코딩
    """
    Mean regularized target encoding based on kfold
    """
    train_new = train.copy()
    test_new = test.copy()
    kf = KFold(n_splits=folds, random_state=1, shuffle=True)
    for col in cols_encode:
        global_mean = train_new[target].mean()
        for train_index, test_index in kf.split(train):
            mean_target = train_new.iloc[train_index].groupby(col)[target].mean()
            train_new.loc[test_index, col + "_mean_enc"] = train_new.loc[test_index, col].map(mean_target)
        train_new[col + "_mean_enc"].fillna(global_mean, inplace=True)
        # making test encoding using full training data
        col_mean = train_new.groupby(col)[target].mean()
        test_new[col + "_mean_enc"] = test_new[col].map(col_mean)
        test_new[col + "_mean_enc"].fillna(global_mean, inplace=True)

    # filtering only mean enc cols
    train_new = train_new.filter(like="mean_enc", axis=1)
    test_new = test_new.filter(like="mean_enc", axis=1)
    return train_new, test_new


def get_larger_range_final(v1, df, num, option=None):
    if option == 'first':
        new_df = get_larger_range_first_utils(v1, df, num)
    elif option == 'last':
        new_df = get_larger_range_last_utils(v1, df, num)
    else:
        new_df = get_larger_range_first_last_utils(v1, df, num)

    return new_df


def get_label_encoding(v1, train_df, test_df):
    label_encoder_v1 = LabelEncoder()
    # train
    v1_info_train = train_df[v1]
    label_encoder_v1.fit(v1_info_train)
    v1_label_encoded_train = label_encoder_v1.transform(v1_info_train)
    v1_label_encoded_train_df = pd.DataFrame(v1_label_encoded_train, columns=['{}_LABEL_ENC'.format(v1)])
    # test - 레이블 인코딩
    v1_info = test_df[v1]
    # 레이블 인코딩 적용(문자 데이터 -> 숫자 데이터)
    v1_label_encoded_test = label_encoder_v1.transform(v1_info)
    v1_label_encoded_test_df = pd.DataFrame(v1_label_encoded_test, columns=['{}_LABEL_ENC'.format(v1)])

    return v1_label_encoded_train_df, v1_label_encoded_test_df


def get_final_train_train(train_df, train_one_hot_enc, test_one_hot_enc, train_target_enc, test_target_enc,
                          train_label_enc_v1,
                          test_label_enc_v1):
    train = pd.concat([train_one_hot_enc, train_target_enc], axis=1)
    train = pd.concat([train, train_label_enc_v1], axis=1)
    data = pd.concat([train, train_df[['TARGET']]], axis=1)
    target = data[['TARGET']]
    data = data.drop(['TARGET'], axis=1)

    test = pd.concat([test_one_hot_enc, test_target_enc], axis=1)
    test = pd.concat([test, test_label_enc_v1], axis=1)
    return data, target, test


def get_add_fe(train, test, add_train_fe, add_test_fe):
    train = pd.concat([train, add_train_fe], axis=1)
    test = pd.concat([test, add_test_fe], axis=1)

    return train, test
