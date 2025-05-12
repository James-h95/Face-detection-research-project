import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class HOG:
    def __init__(self, cell_size=8, block_size=2, bins=9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins
        self.kernelx = np.array([[0,0,0],[-1,0,1],[0,0,0]])
        self.kernely = np.array([[0,-1,0],[0,0,0],[0,1,0]])
        self.angle_unit = 180 / self.bins

    def load_image_as_gray_matrix(self, image_path):
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError(f"can't read image: {image_path}")
        self.img = cv2.resize(self.img, (64, 128))
        self.img = self.img.astype(np.float32) 
        return self.img

    def histogram(self, window):
        height, width = window.shape
        numCellX = width // self.cell_size
        numCellY = height // self.cell_size
        grad_x = cv2.filter2D(self.img, -1, self.kernelx)
        grad_y = cv2.filter2D(self.img, -1, self.kernely)
        rad = np.arctan2(grad_y, grad_x)
        angle_deg = np.degrees(rad)
        angle_deg = np.where(angle_deg < 0, angle_deg + 180, angle_deg)
        hist = np.zeros((numCellX, numCellY, self.bins))
        features = np.zeros(((numCellX- self.block_size + 1)*(numCellY - self.block_size + 1), self.bins * self.block_size * self.block_size))
        for i in range(0, numCellX):
            for j in range(0, numCellY):
                for x in range(i * self.cell_size, i * self.cell_size + self.cell_size):
                    for y in range(j * self.cell_size, j * self.cell_size + self.cell_size):
                        index  =  int(np.floor(angle_deg[y][x] / self.angle_unit))
                        if index == self.bins: 
                            index -= 1
                        hist[i][j][index] += 1
        numFeatureX = numCellX- self.block_size + 1
        numFeatureY = numCellY - self.block_size + 1
        for i in range(numFeatureX):
            for j in range(numFeatureY):
                start = 0 
                for cx in range(0, self.block_size):
                    for cy in range(0, self.block_size):
                        features[i + j * numFeatureX][start : start + self.bins  ] =hist[i + cx][j+cy]   
                        start += self.bins   
                norm = np.linalg.norm(features[i + j * numFeatureX])
                features[ i + j * numFeatureX] /= norm

        return features
    
    def extract_features(self, image_paths):
        """
        Load images and extract HOG features for each image.
        Args:
        - image_paths: List of image file paths.
        
        Returns:
        - features: Extracted HOG features from all images.
        """
        features = []
        for img_path in image_paths:
            img = self.load_image_as_gray_matrix(img_path)
            feature = self.histogram(img)
            features.append(feature)
        return np.array(features)

def train_svm(features, labels):
    """
    Train a SVM classifier using the provided features and labels.
    Args:
    - features: HOG features of the training data.
    - labels: Corresponding labels for each image.
    
    Returns:
    - clf: Trained SVM classifier.
    """
    clf = svm.SVC(kernel='linear', probability = False)  # You can experiment with other kernels like 'rbf'
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
    decision_scores = clf.decision_function(X_test)
    min_v = decision_scores.min()
    max_v = decision_scores.max()
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred, decision_scores

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

