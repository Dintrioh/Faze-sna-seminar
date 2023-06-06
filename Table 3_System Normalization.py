import numpy as np

def subject_Normalization(inputFeatures, indices):
    Number_of_subjects = np.max(indices)
    indices = indices.reshape(-1, 1)
    m, n = inputFeatures.shape

    for i in range(m):
        if np.isnan(inputFeatures[i]).any():
            indices[i] = -1

    newFeatures = inputFeatures.copy()

    for i in range(1, Number_of_subjects + 1):
        sub_idx = np.where(indices[:, 0] == i)[0]
        newFeatures[sub_idx, :-1] = (newFeatures[sub_idx, :-1] - np.mean(newFeatures[sub_idx, :-1], axis=0)) / np.std(newFeatures[sub_idx, :-1], axis=0)

    return newFeatures

def Subject_run_seeds(Exp_Name, numOfSeeds, pdfeatTrainWLabel, pdfeatValWLabel, pdfeatDREAMWLabel, pdfeatTUCDwLabel, Id_of_training_set, Mode):
    collectionTra = []
    collectionVal = []
    collectionDreams = []
    collectionUCD = []

    for ceed in range(1, numOfSeeds + 1):
        resultTra = Subject_getModel_3_classes(Exp_Name + '_Training', ceed, pdfeatTrainWLabel, pdfeatTrainWLabel, Id_of_training_set, 'Tra', 1)
        collectionTra.append(resultTra.T)

        resultVal = Subject_getModel_3_classes(Exp_Name + '_Validation', ceed, pdfeatTrainWLabel, pdfeatValWLabel, Id_of_training_set, 'Val', 1)
        collectionVal.append(resultVal.T)

        resultDreams = Subject_getModel_3_classes(Exp_Name + '_Dreams', ceed, pdfeatTrainWLabel, pdfeatDREAMWLabel, Id_of_training_set, 'Dream', 2)
        collectionDreams.append(resultDreams.T)

        resultUCD = Subject_getModel_3_classes(Exp_Name + '_UCD', ceed, pdfeatTrainWLabel, pdfeatTUCDwLabel, Id_of_training_set, 'UCD', 3)
        collectionUCD.append(resultUCD.T)

    collectionTra = np.array(collectionTra)
    collectionVal = np.array(collectionVal)
    collectionDreams = np.array(collectionDreams)
    collectionUCD = np.array(collectionUCD)

    np.save('./Table_Infos/' + Exp_Name + '_Training.npy', collectionTra)
    np.save('./Table_Infos/' + Exp_Name + '_Validation.npy', collectionVal)
    np.save('./Table_Infos/' + Exp_Name + '_Dreams.npy', collectionDreams)
    np.save('./Table_Infos/' + Exp_Name + '_UCD.npy', collectionUCD)

    result_Tra = np.mean(collectionTra, axis=0)
    result_Val = np.mean(collectionVal, axis=0)
    result_Dreams = np.mean(collectionDreams, axis=0)
    result_UCD = np.mean(collectionUCD, axis=0)

    np.save('./Table_Infos/' + Exp_Name + '_Training_avg.npy', result_Tra)
    np.save('./Table_Infos/' + Exp_Name + '_Validation_avg.npy', result_Val)
    np.save('./Table_Infos/' + Exp_Name + '_Dreams_avg.npy', result_Dreams)
    np.save('./Table_Infos/' + Exp_Name + '_UCD_avg.npy', result_UCD)

def update_subject_indices(features, test_Dataset_Name):
    m, n = features.shape
    output = []
    original_subject_indices = np.load('Original_subject_indices_All.npy')

    if test_Dataset_Name == 'Tra':  # CGMH
        subject_indices = original_subject_indices['Original_subject_indices_Tra']
    elif test_Dataset_Name == 'Val':  # Validation
        subject_indices = original_subject_indices['Original_subject_indices_Val']
    elif test_Dataset_Name == 'Dream':  # Dream
        subject_indices = original_subject_indices['Original_subject_indices_Dream']
    elif test_Dataset_Name == 'UCD':  # UCD
        subject_indices = original_subject_indices['Original_subject_indices_UCD']

    for i in range(m):
        if np.isnan(features[i]).any():
            output.append(i)

    return output

# Usage example:
ftH90NewPS2_CGMH = subject_Normalization(ftH90NewPS2_CGMH, Original_subject_indices_Tra)
ftH90NewPS2_DREAM = subject_Normalization(ftH90NewPS2_DREAM, Original_subject_indices_Dream)
ftH90NewPS2_UCD = subject_Normalization(ftH90NewPS2_UCD, Original_subject_indices_UCD)
ftH90NewPS2_Val = subject_Normalization(ftH90NewPS2_Val, Original_subject_indices_Val)

ftR120NewPS2_CGMH = subject_Normalization(ftR120NewPS2_CGMH, Original_subject_indices_Tra)
ftR120NewPS2_DREAM = subject_Normalization(ftR120NewPS2_DREAM, Original_subject_indices_Dream)
ftR120NewPS2_UCD = subject_Normalization(ftR120NewPS2_UCD, Original_subject_indices_UCD)
ftR120NewPS2_Val = subject_Normalization(ftR120NewPS2_Val, Original_subject_indices_Val)

update_subject_indices(ftH90NewPS2_CGMH, 'Tra')
update_subject_indices(ftH90NewPS2_DREAM, 'Dream')
update_subject_indices(ftH90NewPS2_UCD, 'UCD')
update_subject_indices(ftH90NewPS2_Val, 'Val')
