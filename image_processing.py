import glob
import cv2
import numpy as np


#Transforme une image donnée en une représentation indiquée dans le paramètre "representation"
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


#Renvoie une structure de données avec les images transformées étiquetées
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