import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def getPreprocessPDfeat(pdfeat):
    n, numTS = pdfeat.shape
    pmat = pdfeat.copy()
    n, m = pdfeat.shape
    for i in range(m - 1):
        pmat = pmat[np.where(~np.isnan(pmat[:, i]))[0], :]
        pmat = pmat[np.where(pmat[:, i] < 1e8)[0], :]
    return pmat

def statistics_in_avg(collection):
    m, n = collection.shape
    result = np.zeros((m, 2))
    for i in range(m):
        buff = collection[i, :]
        result[i, 0] = np.mean(buff)
        result[i, 1] = np.std(buff)
    return result

def Subject_getModel(expName, randomSeedNo, pdfeatTrainWLabel, pdfeatValWLabel, id_of_training_dataset, id_of_test_dataset, test_Dataset_Name, MODEL_TYPE):
    numPatientAll = np.load('numPatientAll.npy')
    Original_subject_indices_All = np.load('Original_subject_indices_All.npy')

    if id_of_training_dataset == 1:
        true_label_of_training = 11
    elif id_of_training_dataset == 2:
        true_label_of_training = 5
    else:
        true_label_of_training = 0

    pdfeatT = pdfeatTrainWLabel.copy()
    pdfeatV = pdfeatValWLabel.copy()

    pdfeatT = getPreprocessPDfeat(pdfeatT)
    pdfeatV_buff = pdfeatV.copy()
    pdfeatV_buff = getPreprocessPDfeat(pdfeatV_buff)
    ind_new = update_subject_indices(pdfeatV, test_Dataset_Name)
    if len(ind_new) == 0:
        pdfeatV = pdfeatV_buff
    else:
        m_1, n_1 = pdfeatV.shape
        for i in range(m_1):
            if len(np.where(ind_new[:, 0] == i)[0]) == 0:
                pdfeatV[i, :] = pdfeatV_buff[0, :]
                pdfeatV_buff = pdfeatV_buff[1:, :]
            else:
                pdfeatV[i, :] = np.zeros(n_1)
    
    W = np.where(pdfeatT[:, -1] == true_label_of_training)[0]
    nW = len(W)
    S = np.where(pdfeatT[:, -1] != true_label_of_training)[0]
    nS = len(S)
    pdfeatW = pdfeatT[W, :]
    if MODEL_TYPE == 'SVM':
        fnW = int(1 * nW)
    elif MODEL_TYPE == 'RT':
        fnW = int(1 * nW)
    pdfeatS = pdfeatT[S[np.random.permutation(nS)[:fnW]], :]
    pdfeatT = np.vstack((pdfeatW, pdfeatS))
    
    if test_Dataset_Name == 'Tra':
        numPatient_ = numPatientAll
        Subject_Info = Original_subject_indices_Tra
        Subject_Info[ind_new] = -1
    elif test_Dataset_Name == 'Val':
        numPatient_ = numPatientVal
        Subject_Info = Original_subject_indices_Val
        Subject_Info[ind_new] = -1
    elif test_Dataset_Name == 'Dream':
        numPatient_ = numPatientDream
        Subject_Info = Original_subject_indices_D.
    Subject_Info[ind_new] = -1
    numPatient = numPatient_[Subject_Info >= 0]
    pdfeatV = pdfeatV[Subject_Info >= 0, :]

    Y = pdfeatT[:, -1]
    X = pdfeatT[:, :-1]

    if MODEL_TYPE == 'SVM':
        model = SVC(random_state=randomSeedNo)
    elif MODEL_TYPE == 'RT':
        model = RandomForestClassifier(random_state=randomSeedNo)

    model.fit(X, Y)

    Xv = pdfeatV[:, :-1]
    Yv = pdfeatV[:, -1]
    Yv_hat = model.predict(Xv)

    acc = accuracy_score(Yv, Yv_hat)
    precision = precision_score(Yv, Yv_hat)
    recall = recall_score(Yv, Yv_hat)
    f1 = f1_score(Yv, Yv_hat)

    stats = statistics_in_avg(pdfeatV)
    avg_mean = np.mean(stats[:, 0])
    avg_std = np.mean(stats[:, 1])

    return acc, precision, recall, f1, avg_mean, avg_std
