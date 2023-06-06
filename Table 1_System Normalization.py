import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def getModel_3_classes(expName, randomSeedNo, pdfeatTrainWLabel, pdfeatValWLabel, id_of_training_dataset, id_of_test_dataset):
    # Step 1 :: Load trraining/validation vectors (with labels)
    pdfeatT = pdfeatTrainWLabel
    pdfeatV = pdfeatValWLabel

    # Step 4 :: Extract sleep and awake and rem (with random seed)
    if id_of_training_dataset == 1:  # Dataset provided by CGMH
        W = np.where(pdfeatT[:, -1] == 11)[0]
        R = np.where(pdfeatT[:, -1] == 12)[0]
        S = np.where((pdfeatT[:, -1] != 11) & (pdfeatT[:, -1] != 12))[0]
    elif id_of_training_dataset == 2:  # Dataset "Dreams"
        W = np.where(pdfeatT[:, -1] == 5)[0]
        R = np.where(pdfeatT[:, -1] == 4)[0]
        S = np.where((pdfeatT[:, -1] != 5) & (pdfeatT[:, -1] != 4))[0]
    else:  # Dataset "UCD"
        W = np.where(pdfeatT[:, -1] == 0)[0]
        R = np.where(pdfeatT[:, -1] == 1)[0]
        S = np.where((pdfeatT[:, -1] != 0) & (pdfeatT[:, -1] != 1))[0]

    np.random.seed(randomSeedNo)
    fnW = int(1.1 * len(W))
    rand_per_ind = np.random.permutation(len(S))[:fnW]
    pdfeatW = pdfeatT[W]
    pdfeatR = pdfeatT[R]
    pdfeatS = pdfeatT[S[rand_per_ind]]
    pdfeatT = np.concatenate((pdfeatW, pdfeatR, pdfeatS), axis=0)

    # Step 5 :: Deteremine true label of training dataset
    labelW = (pdfeatT[:, -1] == 11).astype(int)
    labelR = (pdfeatT[:, -1] == 12).astype(int)
    label = 2 * labelW + labelR

    # Label modification for testing dataset
    if id_of_test_dataset == 1:  # CGMH
        TruthLabW = (pdfeatV[:, -1] == 11).astype(int)
        TruthLabR = (pdfeatV[:, -1] == 12).astype(int)
    elif id_of_test_dataset == 2:  # Dreams
        TruthLabW = (pdfeatV[:, -1] == 5).astype(int)
        TruthLabR = (pdfeatV[:, -1] == 4).astype(int)
    else:  # UCD
        TruthLabW = (pdfeatV[:, -1] == 0).astype(int)
        TruthLabR = (pdfeatV[:, -1] == 1).astype(int)

    TruthLab = 2 * TruthLabW + TruthLabR

    # Step 6 :: Generate taining model (for RT only)
    svmModel = SVC()
    svmModel.fit(pdfeatT[:, :-1], label)
    labelV = svmModel.predict(pdfeatV[:, :-1])

    # Step 7 :: Compute the confusion matrix
    confMat = confusion_matrix(TruthLab, labelV)
    tp = np.diag(confMat)
    fp = np.sum(confMat, axis=0) - tp
    fn = np.sum(confMat, axis=1) - tp

    # Step 8 :: Compute the accuracy, precision, recall, and F1 score
    accuracy = np.mean(tp / (tp + fp + fn))
    precision = np.mean(tp / (tp + fp))
    recall = np.mean(tp / (tp + fn))
    f1_score = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score

# Example usage
expName = "Example Experiment"
randomSeedNo = 42
pdfeatTrainWLabel = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
pdfeatValWLabel = np.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
id_of_training_dataset = 1
id_of_test_dataset = 2

accuracy, precision, recall, f1_score = getModel_3_classes(expName, randomSeedNo, pdfeatTrainWLabel, pdfeatValWLabel, id_of_training_dataset, id_of_test_dataset)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
