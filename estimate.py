from sklearn.model_selection import cross_val_score


#Trouve le score moyen du model sur les données d'entraînement grâce à la fonction cross_val_score
def estimate_model_score(train_data, model, k):
    X_train, y_train = train_data[1], train_data[0]

    scores = cross_val_score(model, X_train, y_train, cv=k)
    avg_score = scores.mean()

    return avg_score
