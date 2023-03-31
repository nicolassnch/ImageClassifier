#Prédit la classe d'une seule image
def predict_example_label(example, model):
    label = model.predict([example])
    return label[0]


#Prédit la classe d'un ensemble d'image
def predict_sample_label(data, model):
    predictions_label = []

    for image in data:
        prediction = predict_example_label(image, model)
        predictions_label.append(prediction)

    return predictions_label


#Écrit les prédictions des fichiers dans le répertoire "directory"
#Forme : <nom du fichier> classe
def write_predictions(directory, filename, data, model):
    predictions_label = predict_sample_label(data, model)

    with open(directory + "/Mattéo n'était pas là.txt", "w") as fichier:
        for i in range(len(filename)):
            fichier.write(filename[i] + " " + str(predictions_label[i]) + "\n")
