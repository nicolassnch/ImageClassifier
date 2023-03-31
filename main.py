import glob
import pickle

from estimate import estimate_model_score
from image_processing import load_transform_test_data, load_transform_label_train_data
from predictions import write_predictions
from trainer import fit_with_params

filename = glob.glob("AllTest/*")

print("//////////////// LOADING TRAIN DATA ////////////////")
train_data = load_transform_label_train_data("Data", "PX")

print("//////////////// LOADING TEST DATA ////////////////")
data_Test = load_transform_test_data("AllTest", "PX")

parameters = {
    'algorithm_name': 'SVC',
    'hyperparameters': {
        'C': 10,
        'kernel': 'rbf',
        'gamma': 'scale',
        'verbose': False
    }
}

print("//////////////// LOADING MODEL ////////////////")
##grid_search = search_best_params(train_data, algo_dico_SVC)
##model = fit_with_params(train_data, parameters)
##pickle.dump(model, open('model.pickle', 'wb'))

print("//////////////// LOADING WITH PICKLE ////////////////")
model = pickle.load(open("model.pickle", 'rb'))

print("//////////////// WRITE PREDICTIONS ////////////////")
write_predictions("./", filename, data_Test, model)

##print("//////////////// ESTIMATING ////////////////")
##print(estimate_model_score(train_data, model, 5))
