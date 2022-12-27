from train_utils import run_optuna_xgb, run_optuna_lgbm, run_optuna_cat, run_optuna_tab


def run_model(model=None):
    if model == 'xgboost':
        study = run_optuna_xgb()
    elif model == 'lightgbm':
        study = run_optuna_lgbm()
    elif model == 'catboost':
        study = run_optuna_cat()
    else:
        study = run_optuna_tab()

    return study

