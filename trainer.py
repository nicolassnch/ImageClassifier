from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


#Utilise la fonction GridSearchCV pour trouver et renvoyer les meilleurs hyperparamètres du model choisis
#Le modele est indiqué dans un paramètre "algorithm_name" dans le dictionnaire : "algo_dico"
def search_best_params(train_data, algo_dico):
    X_train = train_data[1]
    Y_train = train_data[0]

    if algo_dico["algorithm_name"] == "GaussianNB":
        gaussian = GaussianNB()

        grid_search = GridSearchCV(gaussian, algo_dico['hyperparameters'], cv=5)
        grid_search.fit(X_train, Y_train)

        return grid_search

    elif algo_dico["algorithm_name"] == "SVC":
        svc = SVC()

        grid_search = GridSearchCV(svc, algo_dico['hyperparameters'], cv=5)
        grid_search.fit(X_train, Y_train)

        return grid_search

    elif algo_dico["algorithm_name"] == "MLP":
        mlp = MLPClassifier()

        grid_search = GridSearchCV(mlp, algo_dico['hyperparameters'], cv=5)
        grid_search.fit(X_train, Y_train)

        return grid_search

    else:
        raise Exception(f"{algo_dico['algorithm_name']} n'est pas un algorithme utilisable")


#À partir des hyperparamètres et du modèle donné dans le dictionnaire algo_dico
#lance l'entraînement du modèle et renvoie le modèle
def fit_with_params(train_data, algo_dico):
    X_train = train_data[1]
    Y_train = train_data[0]

    if algo_dico["algorithm_name"] == "GaussianNB":
        model = GaussianNB(**algo_dico['hyperparameters'])

        model.fit(X_train, Y_train)
        return model

    elif algo_dico["algorithm_name"] == "SVC":
        model = SVC(**algo_dico['hyperparameters'])
        model.fit(X_train, Y_train)
        return model

    elif algo_dico["algorithm_name"] == "MLP":
        model = MLPClassifier(**algo_dico['hyperparameters'])
        model.fit(X_train, Y_train)
        return model

    else:
        raise Exception(f"{algo_dico['algorithm_name']} n'est pas un algorithme utilisable")