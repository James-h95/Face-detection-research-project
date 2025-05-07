import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from HOG import HOG  # 假设你已经有了之前的 HOG 类

def load_data(image_paths, labels):
    """
    Load images and extract HOG features for each image.
    Args:
    - image_paths: List of image file paths.
    - labels: Corresponding labels for the images.
    
    Returns:
    - features: Extracted HOG features from all images.
    - labels: Corresponding labels for each image.
    """
    hog = HOG()
    features = []
    for img_path in image_paths:
        img = hog.load_image_as_gray_matrix(img_path)
        feature = hog.histogram(img)
        features.append(feature)
    return np.array(features), np.array(labels)

def train_svm(features, labels):
    """
    Train a SVM classifier using the provided features and labels.
    Args:
    - features: HOG features of the training data.
    - labels: Corresponding labels for each image.
    
    Returns:
    - clf: Trained SVM classifier.
    """
    clf = svm.SVC(kernel='linear')  # You can experiment with other kernels like 'rbf'
    clf.fit(features, labels)
    return clf

def evaluate_svm(clf, X_test, y_test):
    """
    Evaluate the SVM classifier on test data.
    Args:
    - clf: Trained SVM classifier.
    - X_test: Features of the test data.
    - y_test: True labels of the test data.
    
    Returns:
    - accuracy: Classification accuracy on test data.
    """
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

def predict_with_svm(clf, new_img_path):
    """
    Predict the label of a new image using the trained SVM.
    Args:
    - clf: Trained SVM classifier.
    - new_img_path: Path to the new image.
    
    Returns:
    - prediction: Predicted label for the new image.
    """
    hog = HOG()
    new_img = hog.load_image_as_gray_matrix(new_img_path)
    new_feature = hog.histogram(new_img)
    np_feature = np.array([new_feature])
    np_feature = np_feature.reshape(np_feature.shape[0], -1)

    prediction = clf.predict(np_feature)
    return prediction[0]

def main():
    paths_human = os.listdir("data/MIT")
    paths_human = ["data/MIT/" + g for g in paths_human]
    paths_non_human = os.listdir("data/random_images")
    paths_non_human = ["data/random_images/" + g for g in paths_non_human]
    # Example image paths and labels (Replace these with your own data)
    #image_paths = ["data/MIT/00001_male_back.jpg", "data/MIT/00002_female_back.jpg", "data/MIT/00003_male_side.jpg"]
    train_paths = paths_human[0:250] + paths_non_human[0:250]
    test_paths = paths_human[250:350] + paths_non_human[250:350]

    #random.shuffle(train_paths)

    labels = [1] * 250 + [0] * 250
    
    # Load data and extract features
    features, labels = load_data(train_paths, labels)
    features = features.reshape(features.shape[0], -1)

    test_labels  = [1] * 100 + [0] * 100
    test_features, test_labels = load_data(test_paths, test_labels)
    test_features = test_features.reshape(test_features.shape[0], -1)

    # Split data into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Train SVM classifier
    clf = train_svm(features, labels)
    
    # Evaluate the classifier
    accuracy, pred = evaluate_svm(clf, test_features, test_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    plt.figure(figsize = (12,6))
    plt.plot(pred , label = "pred", color = "blue", linewidth = 1)
    plt.legend()
    plt.show()
    
    # Predict on new image
    new_img_path = "data/MIT/00004_male_back.jpg"  # Replace with a new image path
    prediction = predict_with_svm(clf, new_img_path)
    print(f"Predicted label for new image: {prediction}")

if __name__ == "__main__":
    main()
