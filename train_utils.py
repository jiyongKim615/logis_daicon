from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import optuna
from preprocess import *
import torch
import numpy as np


def run_optuna_xgb():
    train_df, train_v1, test_v1, target_train_new, target_test_new, \
    v1_label_encoded_train_df_3, v1_label_encoded_test_df_3, \
    sgrid_label_encoded_train_df, sgrid_label_encoded_test_df, \
    cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3, \
    cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5 = get_preprocess_all()

    # 최종 데이터 도출
    data, target, test = \
        get_final_train_train(train_df, train_v1, test_v1, target_train_new, target_test_new,
                              v1_label_encoded_train_df_3, v1_label_encoded_test_df_3)

    # 범위 확장 및 label 인코딩 후 얻은 특징 추가
    data, test = get_add_fe(data, test, sgrid_label_encoded_train_df, sgrid_label_encoded_test_df)
    data, test = get_add_fe(data, test, cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3)
    data, test = get_add_fe(data, test, cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5)

    def objective(trial, data=data, target=target):
        train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.15, random_state=42)
        param = {
            'tree_method': 'gpu_hist',
            # this parameter means using the GPU when training our model to speedup the training process
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate',
                                                       [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
            'n_estimators': 10000,
            'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17]),
            'random_state': trial.suggest_categorical('random_state', [2020]),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        }
        model = xgb.XGBRegressor(**param)

        model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)

        preds = model.predict(test_x)

        rmse = mean_squared_error(test_y, preds, squared=False)

        return rmse

    # objective 적용
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return study


def run_optuna_lgbm():
    train_df, train_v1, test_v1, target_train_new, target_test_new, \
    v1_label_encoded_train_df_3, v1_label_encoded_test_df_3, \
    sgrid_label_encoded_train_df, sgrid_label_encoded_test_df, \
    cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3, \
    cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5 = get_preprocess_all()

    # 최종 데이터 도출
    data, target, test = \
        get_final_train_train(train_df, train_v1, test_v1, target_train_new, target_test_new,
                              v1_label_encoded_train_df_3, v1_label_encoded_test_df_3)

    # 범위 확장 및 label 인코딩 후 얻은 특징 추가
    data, test = get_add_fe(data, test, sgrid_label_encoded_train_df, sgrid_label_encoded_test_df)
    data, test = get_add_fe(data, test, cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3)
    data, test = get_add_fe(data, test, cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5)

    def objective(trial, data=data, target=target):
        train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.15, random_state=42)

        param = {
            "objective": "regression",
           # 'device': "gpu",
            "n_jobs": -1,
            "verbose": -1,
            "force_col_wise": True,
            "n_estimators": 1000,
            "boosting_type": 'gbdt',
            "max_bin": 251,
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 1.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 1.0),
            "max_depth": trial.suggest_int("max_depth", 2, 25),
            "num_leaves": trial.suggest_int("num_leaves", 40, 50),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 0.9),
            "subsample": trial.suggest_float("subsample", 0.8, 0.9),
            "subsample_freq": 100,
        }

        model = lgb.LGBMRegressor(**param)

        model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)

        preds = model.predict(test_x)

        rmse = mean_squared_error(test_y, preds, squared=False)

        return rmse

    # objective 적용
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return study


def run_optuna_cat():
    train_df, train_v1, test_v1, target_train_new, target_test_new, \
    v1_label_encoded_train_df_3, v1_label_encoded_test_df_3, \
    sgrid_label_encoded_train_df, sgrid_label_encoded_test_df, \
    cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3, \
    cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5 = get_preprocess_all()

    # 최종 데이터 도출
    data, target, test = \
        get_final_train_train(train_df, train_v1, test_v1, target_train_new, target_test_new,
                              v1_label_encoded_train_df_3, v1_label_encoded_test_df_3)

    # 범위 확장 및 label 인코딩 후 얻은 특징 추가
    data, test = get_add_fe(data, test, sgrid_label_encoded_train_df, sgrid_label_encoded_test_df)
    data, test = get_add_fe(data, test, cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3)
    data, test = get_add_fe(data, test, cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5)

    def objective(trial, data=data, target=target):
        train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.15, random_state=42)

        param = {
            "objective": trial.suggest_categorical("objective", ["RMSE"]),
            'task_type':'GPU',
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "3gb",
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        model = CatBoostRegressor(**param)

        model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)

        preds = model.predict(test_x)

        rmse = mean_squared_error(test_y, preds, squared=False)

        return rmse

    # objective 적용
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return study


def run_optuna_tab():
    train_df, train_v1, test_v1, target_train_new, target_test_new, \
    v1_label_encoded_train_df_3, v1_label_encoded_test_df_3, \
    sgrid_label_encoded_train_df, sgrid_label_encoded_test_df, \
    cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3, \
    cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5 = get_preprocess_all()

    # 최종 데이터 도출
    data, target, test = \
        get_final_train_train(train_df, train_v1, test_v1, target_train_new, target_test_new,
                              v1_label_encoded_train_df_3, v1_label_encoded_test_df_3)

    # 범위 확장 및 label 인코딩 후 얻은 특징 추가
    data, test = get_add_fe(data, test, sgrid_label_encoded_train_df, sgrid_label_encoded_test_df)
    data, test = get_add_fe(data, test, cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3)
    data, test = get_add_fe(data, test, cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5)

    def objective(trial):
        X = data.values
        y = target.values
        y = y.reshape(-1, 1)
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Using {}".format(DEVICE))
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_da = trial.suggest_int("n_da", 56, 64, step=4)
        n_steps = trial.suggest_int("n_steps", 1, 3, step=1)
        gamma = trial.suggest_float("gamma", 1., 1.4, step=0.2)
        n_shared = trial.suggest_int("n_shared", 1, 3)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
        tabnet_params = dict(n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                             lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                             optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                             mask_type=mask_type, n_shared=n_shared,
                             scheduler_params=dict(mode="min",
                                                   patience=trial.suggest_int("patienceScheduler", low=3, high=10),
                                                   # changing sheduler patience to be lower than early stopping patience
                                                   min_lr=1e-5,
                                                   factor=0.5, ),
                             scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                             verbose=0,
                             device_name=DEVICE
                             )  # early stopping
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        CV_score_array = []
        for train_index, test_index in kf.split(X):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            regressor = TabNetRegressor(**tabnet_params)
            regressor.fit(X_train=X_train, y_train=y_train,
                          eval_set=[(X_valid, y_valid)],
                          patience=trial.suggest_int("patience", low=15, high=30),
                          max_epochs=trial.suggest_int('epochs', 1, 100),
                          eval_metric=['rmse'])
            CV_score_array.append(regressor.best_cost)
        avg = np.mean(CV_score_array)
        return avg

    # objective 적용
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return study


def run_optuna(train_df, train_v1, test_v1, target_train_new, target_test_new,
               v1_label_encoded_train_df_3, v1_label_encoded_test_df_3,
               sgrid_label_encoded_train_df, sgrid_label_encoded_test_df,
               cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3,
               cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5):
    # 최종 데이터 도출
    data, target, test = \
        get_final_train_train(train_df, train_v1, test_v1, target_train_new, target_test_new, \
                              v1_label_encoded_train_df_3, v1_label_encoded_test_df_3)

    # 범위 확장 및 label 인코딩 후 얻은 특징 추가
    data, test = get_add_fe(data, test, sgrid_label_encoded_train_df, sgrid_label_encoded_test_df)
    data, test = get_add_fe(data, test, cgrid_label_encoded_train_df_3, cgrid_label_encoded_test_df_3)
    data, test = get_add_fe(data, test, cgrid_label_encoded_train_df_5, cgrid_label_encoded_test_df_5)

    def objective(trial, data=data, target=target):
        train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.15, random_state=42)
        param = {
            'tree_method': 'gpu_hist',
            # this parameter means using the GPU when training our model to speedup the training process
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'learning_rate': trial.suggest_categorical('learning_rate',
                                                       [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
            'n_estimators': 10000,
            'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17]),
            'random_state': trial.suggest_categorical('random_state', [2020]),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        }
        model = xgb.XGBRegressor(**param)

        model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)

        preds = model.predict(test_x)

        rmse = mean_squared_error(test_y, preds, squared=False)

        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return study


def get_optuna_parameter_graph(study):
    # Visualize parameter importances.
    optuna.visualization.plot_param_importances(study)


def get_optuna_best_params(study):
    Best_trial = study.best_trial.params
    return Best_trial
