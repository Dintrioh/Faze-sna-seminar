import numpy as np
from sklearn.svm import SVC

def getPreprocessPDfeat(pdfeat):
    pmat = pdfeat[~np.isnan(pdfeat).any(axis=1)]
    pmat = pmat[pmat[:, :-1] < 1e8]
    return pmat

def statistics_in_avg(collection):
    result = np.zeros((collection.shape[0], 2))
    for i in range(collection.shape[0]):
        buff = collection[i]
        result[i, 0] = np.mean(buff)
        result[i, 1] = np.std(buff)
    return result

def Subject_getModel_3_classes(expName, randomSeedNo, pdfeatTrainWLabel, pdfeatValWLabel, id_of_training_dataset, test_Dataset_Name, id_of_test_dataset):
    numPatientAll = np.load('numPatientAll.npy')
    Original_subject_indices_All = np.load('Original_subject_indices_All.npy')

    pdfeatT = pdfeatTrainWLabel.copy()
    pdfeatV = pdfeatValWLabel.copy()

    pdfeatT = getPreprocessPDfeat(pdfeatT)
    pdfeatV_buff = pdfeatV.copy()
    pdfeatV_buff = getPreprocessPDfeat(pdfeatV_buff)
    ind_new = update_subject_indices(pdfeatV, test_Dataset_Name)

    m_1, n_1 = pdfeatV.shape
    for i in range(m_1):
        if i+1 not in ind_new[:, 0]:
            pdfeatV[i] = pdfeatV_buff[0]
            pdfeatV_buff = pdfeatV_buff[1:]
        else:
            pdfeatV[i] = np.zeros(n_1)

    if id_of_training_dataset == 1:
        W = np.where(pdfeatT[:, -1] == 11)[0]
        nW = len(W)
        R = np.where(pdfeatT[:, -1] == 12)[0]
        nR = len(R)
        S = np.where((pdfeatT[:, -1] != 11) & (pdfeatT[:, -1] != 12))[0]
        nS = len(S)
    elif id_of_training_dataset == 2:
        W = np.where(pdfeatT[:, -1] == 5)[0]
        nW = len(W)
        R = np.where(pdfeatT[:, -1] == 4)[0]
        nR = len(R)
        S = np.where((pdfeatT[:, -1] != 5) & (pdfeatT[:, -1] != 4))[0]
        nS = len(S)
    else:
        W = np.where(pdfeatT[:, -1] == 0)[0]
        nW = len(W)
        R = np.where(pdfeatT[:, -1] == 1)[0]
        nR = len(R)
        S = np.where((pdfeatT[:, -1] != 0) & (pdfeatT[:, -1] != 1))[0]
        nS = len(S)

    np.random.seed(randomSeedNo)
    rand_per_ind = np.random.permutation(nS)
    fnW = int(1.1 * nW)
    pdfeatS = pdfeatT[S[rand_per_ind[:fnW]]]

    pdfeatT = np.concatenate((pdfeatT[W], pdfeatT[R], pdfeatS), axis=0)

    if id_of_training_dataset == 1:
        labelW = pdfeatT[:, -1] == 11
        labelR = pdfeatT[:, -1] == 12
    elif id_of_training_dataset == 2:
        labelW = pdfeatT[:, -1] == 5
        labelR = pdfeatT[:, -1] == 4
    else:
        labelW = pdfeatT[:, -1] == 0
        labelR = pdfeatT[:, -1] == 1

    labelT = np.concatenate((labelW, labelR), axis=0)
    pdfeatT = pdfeatT[:, :-1]

    labelTrain = np.concatenate((labelW, labelR), axis=0)

    labelTrain = np.asarray(labelTrain, dtype=np.int32)
    labelTest = np.asarray(pdfeatV[:, -1] == id_of_test_dataset, dtype=np.int32)
    pdfeatV = pdfeatV[:, :-1]

    pdfeatV = np.asarray(pdfeatV, dtype=np.float32)
    pdfeatT = np.asarray(pdfeatT, dtype=np.float32)

    if expName == 'svm':
        classifier = SVC(kernel='linear', C=1, random_state=42)
        classifier.fit(pdfeatT, labelTrain)
        predictedLabel_val = classifier.predict(pdfeatV)

    if expName == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(random_state=42)
        classifier.fit(pdfeatT, labelTrain)
        predictedLabel_val = classifier.predict(pdfeatV)

    return predictedLabel_val

def update_subject_indices(pdfeatV, test_Dataset_Name):
    ind_new = np.zeros((pdfeatV.shape[0], 1), dtype=int)
    for i in range(pdfeatV.shape[0]):
        ind_new[i, 0] = int(pdfeatV[i, -1])
    return ind_new

# Example usage
expName = 'svm'
randomSeedNo = 42
pdfeatTrainWLabel = np.load('pdfeatTrainWLabel.npy')
pdfeatValWLabel = np.load('pdfeatValWLabel.npy')
id_of_training_dataset = 1
test_Dataset_Name = 'dataset_name'
id_of_test_dataset = 1

predictedLabel_val = Subject_getModel_3_classes(expName, randomSeedNo, pdfeatTrainWLabel, pdfeatValWLabel, id_of_training_dataset, test_Dataset_Name, id_of_test_dataset)

print(predictedLabel_val)
