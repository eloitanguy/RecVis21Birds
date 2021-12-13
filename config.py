CONFIG = {
    'input_size': 384,
    'lr': 0.003,
    # 'scheduler': {
    #     'name': 'exp',
    #     'gamma': 0.95,
    # },
    'scheduler': {
        'name': 'cosWR',
        'T0': 20,
        'T_mult': 2
    },
    'batch_size': 8,
    'epochs': 20,
    'wd': 0.003,
    'augment': True,  # parameters manually set in data.py
    'experiment': 'vit9',
    'model_name': 'vit_large_patch16_384',
    'model': 'vit',
    'linear_dropout': 0.2,
    'RDS_its': 1,  # if >1, operates a variant of K-fold cross-validation /!\ change the last line of main.py for that
    'weighted_sampling': True  # argument does nothing, code modification in main.init_data_... functions
}

XGBOOST_CONFIG = {
    'input_size': 128,
    'max_depth': 4,
    'colsample_bytree': 0.3,
    'n_estimators': 100,
    'learning_rate': 0.3,
    'subsample': 0.2,
    'lambda': 10,
    'TA': -1,
    'VA': -1,
    'experiment': 'alexnetBB_0'
}