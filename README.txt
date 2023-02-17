Libraries utilisées = sklearn, openCV, glob(lecture de repertoire) , numpy

Représentation utilisée : niveau de gris (valeur à passer en paramètres : "GRAY")

Deux algorithmes utilisé :
- Algorithme SVC : (valeur : algo_dico_SVC)
- GaussianNB : (valeur : algo_dico_Gaussian)

structures de données pour les train_data :
    un tuple avec deux listes (les indices des deux listes correspondent) :
        une pour les data X_train : matrice des nuances de gris d'une image
        une pour les labels y_train : liste des "label" (-1 ou 1) pour chaque image

Nous avons dû retirer queqlues images du projets qui ne fonctionnaient pas :
"ll9944.png", "z3tt.png", "ar54ff.png", "aq9ooh.png"

Crédits pour chaque fonction :

Remarque : Nous avons travaillé en groupe tout le long, la répartition du travail sur les fonctions est surtout
    pour la rédaction des fonctions.

Nicolas SANCHEZ : load_transform_label_train_data(), learn_model_from_data(), estimate_model_score()
Yacine TALHAOUI : raw_image_to_representation(), predict_example_label(), write_predictions()
Matteo LIZERO : load_transform_test_data(), predict_sample_label(), main()  .
