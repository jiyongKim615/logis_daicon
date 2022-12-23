import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def get_metric_test_df(data, test, target, Best_trial):
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

    return rmse