import glob
import numpy as np
import cv2

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def raw_image_to_representation(image, representation):
    img = cv2.imread(image)

    desired_size = (500, 500)

    resized_image = cv2.resize(img, desired_size)

    if representation == "HC":

        hist, bins = np.histogram(resized_image.ravel(), 256, [0, 256])

        histogram_to_list = hist.tolist()

        return histogram_to_list

    elif representation == "PX":

        img_array = np.array(resized_image)

        return np.ravel(img_array).tolist()

    elif representation == "GC":

        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        return np.ravel(gray).tolist()

    else:
        raise Exception(f"{representation} n'est pas une représentation correcte")


def load_transform_label_train_data(directory, representation):
    label = []
    data = []

    mer_images = glob.glob(directory + "/Mer/*")
    ailleurs_images = glob.glob(directory + "/Ailleurs/*")

    for path in mer_images:
        label.append(1)
        data.append(raw_image_to_representation(path, representation))

    for path in ailleurs_images:
        label.append(-1)
        data.append(raw_image_to_representation(path, representation))

    return label, data


def load_transform_test_data(directory, representation):
    test_data = []
    test_images = glob.glob(directory + "/*")

    for path in test_images:
        test_data.append(raw_image_to_representation(path, representation))

    return test_data


def learn_model_from_data(train_data, algo_dico):
    X_train = train_data[1]
    Y_train = train_data[0]

    if algo_dico["algorithm_name"] == "GaussianNB":
        gaussian = GaussianNB()

        grid_search = GridSearchCV(gaussian, algo_dico['hyperparameters'], cv=5)
        grid_search.fit(X_train, Y_train)

        model = GaussianNB(**grid_search.best_params_)

        model.fit(X_train, Y_train)
        return model

    elif algo_dico["algorithm_name"] == "SVC":
        svc = SVC()

        grid_search = GridSearchCV(svc, algo_dico['hyperparameters'], cv=5)
        grid_search.fit(X_train, Y_train)

        model = SVC(**grid_search.best_params_)
        model.fit(X_train, Y_train)
        return model

    elif algo_dico["algorithm_name"] == "MLP":
        mlp = MLPClassifier()

        grid_search = GridSearchCV(mlp, algo_dico['hyperparameters'], cv=5)
        grid_search.fit(X_train, Y_train)

        model = MLPClassifier(**grid_search.best_params_)
        model.fit(X_train, Y_train)
        return model

    else:
        raise Exception(f"{algo_dico} n'est pas un algorithme utilisable")


def predict_example_label(example, model):
    label = model.predict([example])
    return label[0]


def predict_sample_label(data, model):
    predictions_label = []

    for image in data:
        prediction = predict_example_label(image, model)
        predictions_label.append(prediction)

    return predictions_label


def write_predictions(directory, filename, data, model):
    predictions_label = predict_sample_label(data, model)

    with open(directory + "/prediction.txt", "w") as fichier:
        for i in range(len(filename)):
            fichier.write(filename[i] + " " + str(predictions_label[i]) + "\n")


def estimate_model_score(train_data, model, k):
    X_train, y_train = train_data[1], train_data[0]

    scores = cross_val_score(model, X_train, y_train, cv=k)
    avg_score = scores.mean()

    return avg_score


if __name__ == '__main__':
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
            'verbose': [False]
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

    print("//////////////// LOADING MODEL ////////////////")
    model = learn_model_from_data(train_data, algo_dico_SVC)
    ##write_predictions("./", filename, data_Test, model)

    print("//////////////// ESTIMATING ////////////////")
    ##model = GaussianNB(**algo_dico_Gaussian["hyperparameters"])
    print(estimate_model_score(train_data, model, 8))

#SAUVEGARDER LES MODELS AVEC PICKLE