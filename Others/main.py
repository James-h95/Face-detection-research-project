from SVM import *
from DRAW import *
from joblib import dump

def get_paths(dir):
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory '{dir}' does not exist.")
    data = os.listdir(dir)
    data = [os.path.join(dir, g) for g in data]
    return data

def run(hog):
    paths_train_human = get_paths("Others/sample_data/training_set/human")
    paths_train_nonHuman = get_paths("Others/sample_data/training_set/nonHuman")

    paths_test_human = get_paths("Others/sample_data/testing_set/human")
    paths_test_nonHuman = get_paths("Others/sample_data/testing_set/nonHuman")

    train_paths = paths_train_human + paths_train_nonHuman
    test_paths = paths_test_human+ paths_test_nonHuman

    train_labels = np.array([1] * len(paths_train_human) + [0] * len(paths_train_nonHuman))
    test_labels  = np.array([1] * len(paths_test_human) + [0] * len(paths_test_nonHuman))
    
    # Load data and extract features
    features = hog.extract_features(train_paths)
    features = features.reshape(features.shape[0], -1)
    test_features = hog.extract_features(test_paths)
    test_features = test_features.reshape(test_features.shape[0], -1)
    
    # Train SVM classifier
    clf = train_svm(features, train_labels)
    
    # Evaluate the classifier on testing set
    accuracy, pred, scores = evaluate_svm(clf, test_features, test_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    wrongs = []
    for i in range(len(test_labels)):
        if pred[i]!=test_labels[i]:
            wrongs.append(i)
    print("wrong_indexes:", wrongs)

    # plt.figure(figsize = (12,6))
    # plt.plot(pred , label = "pred", color = "blue", linewidth = 1)
    # plt.legend()
    # plt.show()

    return scores, test_labels, clf

if __name__ == "__main__":
    hog = HOG(cell_size = 8,block_size = 2, bins = 9)
    scores1, labels1, clf = run(hog)
    dump(clf, 'svm_model.joblib')

    hog = HOG(cell_size = 16,block_size = 2, bins = 9)
    scores2, labels2, _ = run(hog)

    hog = HOG(cell_size = 8,block_size = 2, bins = 6)
    scores3, labels3, _ = run(hog)

    hog = HOG(cell_size = 8,block_size = 2, bins = 12)
    scores4, labels4, _ = run(hog)

    plot_multiple_fppw_curves([(scores1,labels1),(scores2,labels2),(scores3,labels3),(scores4,labels4)])

    print("end")