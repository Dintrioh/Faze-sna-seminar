import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def getPreprocessPDfeat(pdfeat):
    pmat = pdfeat.copy()
    for i in range(pmat.shape[1] - 1):
        pmat = pmat[~np.isnan(pmat[:, i]), :]
        pmat = pmat[pmat[:, i] < 1e8, :]
    return pmat

def statistics_in_avg(collection):
    return np.mean(collection, axis=0), np.std(collection, axis=0)

def Subject_getModel(expName, randomSeedNo, pdfeatTrainWLabel, pdfeatValWLabel,
                     id_of_training_dataset, id_of_test_dataset, test_Dataset_Name, MODEL_TYPE):
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
    pdfeatV_buff = getPreprocessPDfeat(pdfeatV)
    
    ind_new = update_subject_indices(pdfeatV, test_Dataset_Name)
    if len(ind_new) == 0:
        pdfeatV = pdfeatV_buff
    else:
        m_1, n_1 = pdfeatV.shape
        for i in range(m_1):
            if i + 1 not in ind_new[:, 0]:
                pdfeatV[i] = pdfeatV_buff[0]
                pdfeatV_buff = pdfeatV_buff[1:]
            else:
                pdfeatV[i] = np.zeros(n_1)
    
    W = np.where(pdfeatT[:, -1] == true_label_of_training)[0]
    nW = len(W)
    S = np.where(pdfeatT[:, -1] != true_label_of_training)[0]
    nS = len(S)
    pdfeatW = pdfeatT[W]
    
    if MODEL_TYPE == 'SVM':
        fnW = int(np.floor(1 * nW))
    elif MODEL_TYPE == 'RT':
        fnW = int(np.floor(1 * nW))
    
    np.random.seed(randomSeedNo)
    rand_per_ind = np.random.permutation(nS)
    
    pdfeatS = pdfeatT[S[rand_per_ind[:fnW]]]
    pdfeatT = np.vstack((pdfeatW, pdfeatS))
    
    if test_Dataset_Name == 'Tra':
        numPatient_ = numPatientAll
        Subject_Info = Original_subject_indices_Tra.copy()
        Subject_Info[ind_new] = -1
    elif test_Dataset_Name == 'Val':
        numPatient_ = numPatientVal
        Subject_Info = Original_subject_indices_Val.copy()
        Subject_Info[ind_new] = -1
    elif test_Dataset_Name == 'Dream':
        numPatient_ = numPatientDream
        Subject_Info = Original_subject_indices_Dream.copy()
        Subject_Info[ind_new] = -1
    elif test_Dataset_Name == 'UCD':
        numPatient_ = numPatientUCD
        Subject_Info = Original_subject_indices_UCD.copy()
        Subject_Info[ind_new] = -1
    
    m, n = num.
    m, n = numPatient_.shape

    MODEL_FEATURE_NUM = pdfeatT.shape[1] - 1

    TRAIN_PATIENT_NUM = pdfeatT.shape[0]
    VAL_PATIENT_NUM = pdfeatV.shape[0]

    num_C = 30
    num_G = 20
    CV_rbf = np.zeros((num_C, num_G))
    Val_rbf = np.zeros((num_C, num_G))
    subject_set = np.zeros((num_C, num_G, m))

    for k in range(1, m + 1):
        numPatients = numPatient_[k - 1, 0]
        Subject_Info_ = Subject_Info[Subject_Info[:, 0] == k, :]
        Subject_Info_ = Subject_Info_[:, 1]

        pdfeatW = pdfeatT[np.isin(pdfeatT[:, -1], Subject_Info_), :]
        W = np.where(pdfeatW[:, -1] == true_label_of_training)[0]
        pdfeatS = pdfeatT[np.isin(pdfeatT[:, -1], Subject_Info_) == False, :]

        for i in range(num_C):
            C_pow = -3 + i * 2
            C = 10 ** C_pow
            for j in range(num_G):
                G_pow = -13 + j * 2
                G = 10 ** G_pow

                if MODEL_TYPE == 'SVM':
                    clf = SVC(kernel='rbf', C=C, gamma=G, probability=True)
                elif MODEL_TYPE == 'RT':
                    clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2,
                                                 random_state=0)

                clf.fit(pdfeatW[:, :-1], pdfeatW[:, -1])

                if MODEL_TYPE == 'SVM':
                    probS = clf.predict_proba(pdfeatS[:, :-1])
                elif MODEL_TYPE == 'RT':
                    probS = clf.predict_proba(pdfeatS[:, :-1])[:, 1]

                if MODEL_TYPE == 'SVM':
                    probT = clf.predict_proba(pdfeatV[:, :-1])
                elif MODEL_TYPE == 'RT':
                    probT = clf.predict_proba(pdfeatV[:, :-1])[:, 1]

                Subject_pred_label = np.zeros(numPatients)
                Subject_pred_prob = np.zeros(numPatients)
                Subject_true_label = np.zeros(numPatients)

                for j in range(numPatients):
                    Subject_pred_prob[j] = np.mean(probT[Subject_Info_ == (j + 1)])
                    if Subject_pred_prob[j] > 0.5:
                        Subject_pred_label[j] = 1
                    else:
                        Subject_pred_label[j] = 0
                    Subject_true_label[j] = pdfeatV[Subject_Info_ == (j + 1), -1][0]

                CV_rbf[i, j] = roc_auc_score(Subject_true_label, Subject_pred_prob)
                Val_rbf[i, j] = np.mean(probT)

                subject_set[i, j, k - 1] = roc_auc_score(Subject_true_label, Subject_pred_prob)

    max_idx = np.unravel_index(np.argmax(CV_rbf), CV_rbf.shape)
    optC = 10 ** (-3 + max_idx[0] * 2)
    optG = 10 ** (-13 + max_idx[1] * 2)

    if MODEL_TYPE == 'SVM':
        clf = SVC(kernel='rbf', C=optC, gamma=optG, probability=True)
    elif MODEL_TYPE == 'RT':
        clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)

    clf.fit(pdfeatT[:, :-1], pdfeatT[:, -1])

    if MODEL_TYPE == 'SVM':
        probT = clf.predict_proba(pdfeatT[:, :-1])
        probV = clf.predict_proba(pdfeatV[:, :-1])
        probS = clf.predict_proba(pdfeatS[:, :-1])
    elif MODEL_TYPE == 'RT':
        probT = clf.predict_proba(pdfeatT[:, :-1])[:, 1]
        probV = clf.predict_proba(pdfeatV[:, :-1])[:, 1]
        probS = clf.predict_proba(pdfeatS[:, :-1])[:, 1]

    Subject_pred_label_T = np.zeros(TRAIN_PATIENT_NUM)
    Subject_pred_prob_T = np.zeros(TRAIN_PATIENT_NUM)
    Subject_true_label_T = np.zeros(TRAIN_PATIENT_NUM)

    for k in range(1, m + 1):
        numPatients = numPatient_[k - 1, 0]
        Subject_Info_ = Subject_Info[Subject_Info[:, 0] == k, :]
        Subject_Info_ = Subject_Info_[:, 1]

        Subject_pred_prob_T[Subject_Info_ - 1] = np.mean(probT[Subject_Info_ - 1])
        Subject_true_label_T[Subject_Info_ - 1] = pdfeatT[Subject_Info_ - 1, -1]

    auc_T = roc_auc_score(Subject_true_label_T, Subject_pred_prob_T)

    Subject_pred_label_V = np.zeros(VAL_PATIENT_NUM)
    Subject_pred_prob_V = np.zeros(VAL_PATIENT_NUM)
    Subject_true_label_V = np.zeros(VAL_PATIENT_NUM)

    for k in range(1, m + 1):
        numPatients = numPatient_[k - 1, 0]
        Subject_Info_ = Subject_Info[Subject_Info[:, 0] == k, :]
        Subject_Info_ = Subject_Info_[:, 1]

        Subject_pred_prob_V[Subject_Info_ - 1] = np.mean(probV[Subject_Info_ - 1])
        Subject_true_label_V[Subject_Info_ - 1] = pdfeatV[Subject_Info_ - 1, -1]

    auc_V = roc_auc_score(Subject_true_label_V, Subject_pred_prob_V)

    Subject_pred_label_S = np.zeros(pdfeatS.shape[0])
    Subject_pred_prob_S = np.zeros(pdfeatS.shape[0])

    for j in range(pdfeatS.shape[0]):
        Subject_pred_prob_S[j] = np.mean(probS[Subject_Info_ == (j + 1)])
        if Subject_pred_prob_S[j] > 0.5:
            Subject_pred_label_S[j] = 1
        else:
            Subject_pred_label_S[j] = 0

    return auc_T, auc_V, Subject_pred_label_T, Subject_pred_prob_T, Subject_true_label_T, \
           Subject_pred_label_V, Subject_pred_prob_V, Subject_true_label_V, \
           Subject_pred_label_S, Subject_pred_prob_S
