import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def get_preprocess_pdfeat(pdfeat):
    pmat = pdfeat.copy()
    n, m = pdfeat.shape
    for i in range(m - 1):
        pmat = pmat[~np.isnan(pmat[:, i])]
        pmat = pmat[pmat[:, i] < 1e8]
    return pmat

def subject_get_model(expName, randomSeedNo, pdfeatTrainWLabel, pdfeatValWLabel,
                      id_of_training_dataset, id_of_test_dataset, test_Dataset_Name, MODEL_TYPE):
    numPatientAll = np.load('numPatientAll.npy')
    Original_subject_indices_All = np.load('Original_subject_indices_All.npy')

    if id_of_training_dataset == 1:
        true_label_of_training = 11
    elif id_of_training_dataset == 2:
        true_label_of_training = 5
    else:
        true_label_of_training = 0

    pdfeatT = get_preprocess_pdfeat(pdfeatTrainWLabel)
    pdfeatV = get_preprocess_pdfeat(pdfeatValWLabel)
    ind_new = update_subject_indices(pdfeatV, test_Dataset_Name)

    if len(ind_new) == 0:
        pdfeatV = pdfeatV_buff
    else:
        m_1, n_1 = pdfeatV.shape
        for i in range(m_1):
            if i+1 not in ind_new[:, 0]:
                pdfeatV[i, :] = pdfeatV_buff[0, :]
                pdfeatV_buff = pdfeatV_buff[1:, :]

    W = np.where(pdfeatT[:, -1] == true_label_of_training)[0]
    nW = len(W)
    S = np.where(pdfeatT[:, -1] != true_label_of_training)[0]
    nS = len(S)
    pdfeatW = pdfeatT[W, :]
    np.random.seed(randomSeedNo)
    rand_per_ind = np.random.permutation(nS)

    if MODEL_TYPE == 'SVM':
        fnW = int(np.floor(1 * nW))
    elif MODEL_TYPE == 'RT':
        fnW = int(np.floor(1 * nW))

    pdfeatS = pdfeatT[S[rand_per_ind[:fnW]], :]
    pdfeatT = np.concatenate((pdfeatW, pdfeatS), axis=0)

    label = pdfeatT[:, -1] == true_label_of_training

    switcher = {
        1: (numPatientAll, Original_subject_indices_Tra),
        2: (numPatientDream, Original_subject_indices_Dream),
        3: (numPatientUCD, Original_subject_indices_UCD)
    }
    numPatient_, Subject_Info = switcher.get(id_of_test_dataset)

    result = []
    AUC = []

    for i in range(numPatient_.shape[0]):
        ind = Subject_Info[:, 0] == (i+1)
        TruthLab_Buff = TruthLab[ind, :]
        labelV_Buff = labelV[ind, :]
        AUC = scoreV[ind, 1]

        TP = np.sum(np.logical_and(TruthLab_Buff, labelV_Buff))
        TN = np.sum(np.logical_and(~TruthLab_Buff, ~labelV_Buff))
        FN = np.sum(TruthLab_Buff) - TP
        FP = np.sum(~TruthLab_Buff) - TN
        sen = TP / (TP + FN)
        spe = TN / (TN + FP)

        result.append([sen, spe, AUC])

    return result

def statistics_in_avg(result):
    result_avg = np.mean(result, axis=0)
    return result_avg

# Load the feature vectors
data = np.load('pdfeatNewPS2.npy', allow_pickle=True)
pdfeatNewPS2 = data.item()

# Define the datasets and their indices
numPatientAll = np.load('numPatientAll.npy')
Original_subject_indices_Tra = np.load('Original_subject_indices_All.npy')
numPatientDream = np.load('numPatientDream.npy')
Original_subject_indices_Dream = np.load('Original_subject_indices_Dream.npy')
numPatientUCD = np.load('numPatientUCD.npy')
Original_subject_indices_UCD = np.load('Original_subject_indices_UCD.npy')

# Define the feature matrices
pdfeatTrainWLabel = pdfeatNewPS2['pdfeatTrainWLabel']
pdfeatValWLabel = pdfeatNewPS2['pdfeatValWLabel']
pdfeatTestWLabel = pdfeatNewPS2['pdfeatTestWLabel']
pdfeatTestNoLabel = pdfeatNewPS2['pdfeatTestNoLabel']

# Normalize the loaded feature vectors based on subject indices
pdfeatTrainWLabel = get_preprocess_pdfeat(pdfeatTrainWLabel)
pdfeatValWLabel = get_preprocess_pdfeat(pdfeatValWLabel)
pdfeatTestWLabel = get_preprocess_pdfeat(pdfeatTestWLabel)
pdfeatTestNoLabel = get_preprocess_pdfeat(pdfeatTestNoLabel)

# Define the label vectors
TruthLab = pdfeatNewPS2['TruthLab']
labelV = pdfeatNewPS2['labelV']
scoreV = pdfeatNewPS2['scoreV']

# Define the parameters
randomSeedNo = 1
test_Dataset_Name = "PS2"
MODEL_TYPE = "SVM"

# Perform experiments using different datasets
result1 = subject_get_model("All", randomSeedNo, pdfeatTrainWLabel, pdfeatValWLabel, 1, 1, test_Dataset_Name, MODEL_TYPE)
result2 = subject_get_model("Dream", randomSeedNo, pdfeatTrainWLabel, pdfeatValWLabel, 2, 1, test_Dataset_Name, MODEL_TYPE)
result3 = subject_get_model("UCD", randomSeedNo, pdfeatTrainWLabel, pdfeatValWLabel, 3, 1, test_Dataset_Name, MODEL_TYPE)

# Calculate average statistics
result_avg1 = statistics_in_avg(result1)
result_avg2 = statistics_in_avg(result2)
result_avg3 = statistics_in_avg(result3)
