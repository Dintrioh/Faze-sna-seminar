import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score

def getModel_3_classes(expName, randomCeedNo, pdfeatTrainWLabel, pdfeatValWLabel, id_of_training_dataset, id_of_test_dataset):
    def getPreprocessPDfeat(pdfeat):
        pmat = pdfeat.copy()
        pmat = pmat[~np.isnan(pmat[:, :-1]).any(axis=1)]
        pmat = pmat[pmat[:, :-1] < 1e8]
        return pmat

    def compute_evaluation_metrics(labelV, TruthLab):
        S22 = np.sum(np.logical_and(TruthLab * labelV == 4, labelV == 1))
        S11 = np.sum(np.logical_and(TruthLab * labelV == 1, labelV == 1))
        S00 = np.sum(np.logical_and(TruthLab + labelV == 0, labelV == 0))

        S21 = np.sum(np.logical_and(TruthLab == 2, labelV == 1))
        S20 = np.sum(np.logical_and(TruthLab == 2, labelV == 0))
        S12 = np.sum(np.logical_and(TruthLab == 1, labelV == 2))
        S10 = np.sum(np.logical_and(TruthLab == 1, labelV == 0))
        S02 = np.sum(np.logical_and(TruthLab == 0, labelV == 2))
        S01 = np.sum(np.logical_and(TruthLab == 0, labelV == 1))

        SE2 = S22 / (S20 + S21 + S22)
        SE1 = S11 / (S10 + S11 + S12)
        SE0 = S00 / (S00 + S01 + S02)

        pP2 = S22 / (S02 + S12 + S22)
        pP1 = S11 / (S01 + S11 + S21)
        pP0 = S00 / (S00 + S10 + S20)

        totalSubjects = (S22 + S11 + S00 + S21 + S20 + S12 + S10 + S02 + S01)
        Acc = (S22 + S11 + S00) / totalSubjects
        p_o = Acc
        p_e = ((S20 + S21 + S22) * (S02 + S12 + S22) + (S10 + S11 + S12) * (S01 + S11 + S21) + (S00 + S01 + S02) * (
                    S00 + S10 + S20)) / (totalSubjects * totalSubjects)
        Kappa = (p_o - p_e) / (1 - p_e)

        return [S22, S11, S00, S21, S20, S12, S10, S02, S01, SE2, SE1, SE0, pP2, pP1, pP0, Acc, Kappa]

    pdfeatT = pdfeatTrainWLabel.copy()
    pdfeatV = pdfeatValWLabel.copy()

    pdfeatT = getPreprocessPDfeat(pdfeatT)
    pdfeatV = getPreprocessPDfeat(pdfeatV)

    if id_of_training_dataset == 1:
        W = np.where(pdfeatT[:, -1] == 11)[0]
        nW = len(W)
        R = np.where(pdfeatT[:, -1] == 12.
        R = np.where(pdfeatT[:, -1] == 12)[0]
        nR = len(R)
        N = np.where(pdfeatT[:, -1] == 13)[0]
        nN = len(N)
    elif id_of_training_dataset == 2:
        W = np.where(pdfeatT[:, -1] == 1)[0]
        nW = len(W)
        R = np.where(pdfeatT[:, -1] == 2)[0]
        nR = len(R)
        N = np.where(pdfeatT[:, -1] == 3)[0]
        nN = len(N)
    else:
        raise ValueError("Invalid id_of_training_dataset")

    X_train = pdfeatT[:, :-1]
    y_train = pdfeatT[:, -1]

    X_val = pdfeatV[:, :-1]
    y_val = pdfeatV[:, -1]

    # Selecting only the samples of the specified classes
    if expName == 'WvsR':
        X_train = X_train[np.concatenate((W, R, N)), :]
        y_train = y_train[np.concatenate((W, R, N))]
        X_val = X_val[np.concatenate((W, R, N)), :]
        y_val = y_val[np.concatenate((W, R, N))]
    elif expName == 'WvsN':
        X_train = X_train[np.concatenate((W, R, N)), :]
        y_train = y_train[np.concatenate((W, R, N))]
        X_val = X_val[np.concatenate((W, R, N)), :]
        y_val = y_val[np.concatenate((W, R, N))]
    elif expName == 'RvsN':
        X_train = X_train[np.concatenate((W, R, N)), :]
        y_train = y_train[np.concatenate((W, R, N))]
        X_val = X_val[np.concatenate((W, R, N)), :]
        y_val = y_val[np.concatenate((W, R, N))]
    else:
        raise ValueError("Invalid experiment name")

    # Creating an SVM classifier
    svm = SVC(kernel='linear', random_state=randomCeedNo)

    # Training the SVM classifier
    svm.fit(X_train, y_train)

    # Making predictions on the validation set
    y_pred = svm.predict(X_val)

    # Computing evaluation metrics
    evaluation_metrics = compute_evaluation_metrics(y_pred, y_val)

    return svm, evaluation_metrics
