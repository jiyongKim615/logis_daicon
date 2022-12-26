from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import optuna
from preprocess import get_final_train_train, get_add_fe


# def objective(trial, data=data, target=target):
#     train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.15, random_state=42)
#     param = {
#         'tree_method': 'gpu_hist',
#         # this parameter means using the GPU when training our model to speedup the training process
#         'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
#         'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
#         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
#         'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
#         'learning_rate': trial.suggest_categorical('learning_rate', [0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]),
#         'n_estimators': 10000,
#         'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17]),
#         'random_state': trial.suggest_categorical('random_state', [2020]),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
#     }
#     model = xgb.XGBRegressor(**param)
#
#     model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100, verbose=False)
#
#     preds = model.predict(test_x)
#
#     rmse = mean_squared_error(test_y, preds, squared=False)
#
#     return rmse


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
    Best_trial["n_estimators"], Best_trial["tree_method"] = 10000, 'gpu_hist'
    return Best_trial
