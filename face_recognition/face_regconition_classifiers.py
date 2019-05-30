
# import all classifier sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import math
import os
import os.path
import pickle
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

# initialize name, classifiers methods, extensions, font

names = ["Nearest_Neighbors", "Gaussian_Process",
         "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()
    ]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
font = ImageFont.truetype("arial.ttf", 30);

face_db_path = "./face_db"
test_db_path = "test_db"

def train(train_dir, model_save_path, clf, verbose=False):
    """
    Trains aclassifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: path to save model on disk
    :param clf: specified classifier method
    :param verbose: verbosity of training

    :return: returns classifier that was trained on the given data.
    """

    X = []
    y = []
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    clf.fit(X, y)

    # Save the trained classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(clf, f)

    return clf


def predict(X_img_path, clf=None, model_path=None, distance_threshold=0.4):
    """
    Recognizes faces in given image using a trained classifier

    :param X_img_path: path to image to be recognized
    :param clf: (optional) a  classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled classifier. if not specified, model_save_path must be clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if clf is None and model_path is None:
        raise Exception("Must supply classifier either thourgh clf or model_path")

    # Load a trained  model (if one was passed in)
    print("Model path: ")
    print(model_path)

    if clf is None:
        with open(model_path, 'rb') as f:
            #
            clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the  model to find the best matches for the test face
    print(clf)

    # if hasattr(clf, "kneighbors"):
    #     closest_distances = clf.kneighbors(faces_encodings, n_neighbors=1)

    # if hasattr(clf, "decision_function"):
    #     closest_distances = clf.decision_function(faces_encodings)
    # else:

    closest_distances = clf.predict_proba(faces_encodings)[:, 1]

    print("closest_distances: ")
    print(closest_distances)

    are_matches = [closest_distances[i] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions, clf_name):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        #name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name, font=font)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255), font=font)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    # pil_image.show()
    pil_image.save("output/" + clf_name + "_" + Path(img_path).name)


if __name__ == "__main__":
    # STEP 1: Train the  classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training All classifier...")

    # iterator each classifier sklearn
    for name, clf in zip(names, classifiers):
        model_save_path = "trained_{}_model.clf".format(name)
        classifier = train(face_db_path, model_save_path, clf)
        print("Training {} complete!".format(name))

    # STEP 2: Using the trained classifier, make predictions for unknown images
    for clf_name in names:
        for image_file in os.listdir(test_db_path):
            full_file_path = os.path.join(test_db_path, image_file)

            print("Looking for faces in {}".format(image_file))

            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance

            predictions = predict(full_file_path, model_path="trained_{}_model.clf".format(clf_name))

            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))

            # Display results overlaid on an image
            show_prediction_labels_on_image(os.path.join(test_db_path, image_file), predictions, clf_name)
