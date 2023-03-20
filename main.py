import glob

from image_processing import load_transform_test_data, load_transform_label_train_data
from predictions import write_predictions
from trainer import fit_with_params

filename = glob.glob("test/*")

print("//////////////// LOAD TEST DATA ////////////////")
data_Test = load_transform_test_data("test", "PX")

print("//////////////// LOAD TRAIN DATA ////////////////")
train_data = load_transform_label_train_data("Data", "PX")

algo_dico_SVC = {
    'algorithm_name': 'SVC',
    'hyperparameters': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'gamma': ['scale', 0.1, 1],
        'verbose': [True]
    }
}

algo_dico_Gaussian = {
    'algorithm_name': 'GaussianNB',
    'hyperparameters': {
        'var_smoothing': [1e-9, 1e-7, 1e-5]
    }
}

algo_dico_MLP = {
    'algorithm_name': 'MLP',
    'hyperparameters': {
        'hidden_layer_sizes': [(10,), (20,), (30,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
    }
}

print("//////////////// LOAD MODEL ////////////////")
##grid_search = search_best_params(train_data, algo_dico_SVC)

final_dico_SVC = {
    'algorithm_name': 'SVC',
    'hyperparameters': {
        'C': 1,
        'kernel': 'rbf',
        'gamma': 'scale',
        'verbose': False
    }
}

model = fit_with_params(train_data, final_dico_SVC)

print("//////////////// WRITE PREDICTIONS ////////////////")
write_predictions("./", filename, data_Test, model)

##print("//////////////// ESTIMATING ////////////////")
##model = GaussianNB(**algo_dico_Gaussian["hyperparameters"])
##print(estimate_model_score(train_data, model, 8))
