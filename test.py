import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from preprocess import get_final_train_train, get_preprocess_all


def get_metric_test_df(Best_trial):
    train_df, train_v1, test_v1, target_train_new, target_test_new, \
    v1_label_encoded_train_df_3, v1_label_encoded_test_df_3, \
    sgrid_label_encoded_train_df, sgrid_label_encoded_test_df, \
    cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3, \
    cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5 = get_preprocess_all()

    # 최종 데이터 도출
    data, target, test = \
        get_final_train_train(train_df, train_v1, test_v1, target_train_new, target_test_new,
                              v1_label_encoded_train_df_3, v1_label_encoded_test_df_3)

    columns = data.columns.to_list()
    train = pd.concat([data, target], axis=1)

    preds = np.zeros(test.shape[0])
    kf = KFold(n_splits=5, random_state=48, shuffle=True)
    rmse = []  # list contains rmse for each fold
    n = 0
    for trn_idx, test_idx in kf.split(train[columns], train['TARGET']):
        X_tr, X_val = train[columns].iloc[trn_idx], train[columns].iloc[test_idx]
        y_tr, y_val = train['TARGET'].iloc[trn_idx], train['TARGET'].iloc[test_idx]
        model = xgb.XGBRegressor(**Best_trial)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
        preds += model.predict(test[columns]) / kf.n_splits
        rmse.append(mean_squared_error(y_val, model.predict(X_val), squared=False))
        print(f"fold: {n + 1} ==> rmse: {rmse[n]}")
        n += 1

    return preds, rmse


def gen_submission_df(try_num, preds):
    # 제출 데이터 생성
    submission_df = pd.read_csv('sample_submission.csv')
    submission_df['운송장_건수'] = preds
    # submission_df['운송장_건수'] = round(submission_df['운송장_건수'], 0)

    submission_df.to_csv('submission_{}.csv'.format(try_num), index=False)