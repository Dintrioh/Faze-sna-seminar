import numpy as np

def subject_normalization(pdfeat, original_subject_indices):
    n, m = pdfeat.shape
    pmat = pdfeat.copy()
    for i in range(m - 1):
        pmat = pmat[~np.isnan(pmat[:, i])]
        pmat = pmat[pmat[:, i] < 1e8]
    return pmat

# Load features
v90T = np.concatenate((ftH90NewPS2_CGMH[:, :16], ftH90NewPS2_CGMH[:, -1:]), axis=1)
v90V = np.concatenate((ftH90NewPS2_Val[:, :16], ftH90NewPS2_Val[:, -1:]), axis=1)
v90Dreams = np.concatenate((ftH90NewPS2_DREAM[:, :16], ftH90NewPS2_DREAM[:, -1:]), axis=1)
v90UCD = np.concatenate((ftH90NewPS2_UCD[:, :16], ftH90NewPS2_UCD[:, -1:]), axis=1)

v120T = np.concatenate((ftR120NewPS2_CGMH[:, :16], ftR120NewPS2_CGMH[:, 24:40], ftR120NewPS2_CGMH[:, -1:]), axis=1)
v120V = np.concatenate((ftR120NewPS2_Val[:, :16], ftR120NewPS2_Val[:, 24:40], ftR120NewPS2_Val[:, -1:]), axis=1)
v120Dreams = np.concatenate((ftR120NewPS2_DREAM[:, :16], ftR120NewPS2_DREAM[:, 24:40], ftR120NewPS2_DREAM[:, -1:]), axis=1)
v120UCD = np.concatenate((ftR120NewPS2_UCD[:, :16], ftR120NewPS2_UCD[:, 24:40], ftR120NewPS2_UCD[:, -1:]), axis=1)

# Step 2: Generate columns of the table

# Generate SI Table 3 (training on "Dreams")
expName = 'EXP_Table_SI_3_training_on_Dreams'
numOfSeeds = 1

subject_indices_Dreams = np.arange(1, len(v120Dreams) + 1)
subject_indices_Tra = subject_normalization(v120T[:, :-1], Original_subject_indices_Tra_without_wake)
subject_indices_Val = subject_normalization(v120V[:, :-1], Original_subject_indices_Val_without_wake)
subject_indices_UCD = subject_normalization(v120UCD[:, :-1], Original_subject_indices_UCD_without_wake)

# Training SVM model
# ...

print(f'{expName} is Done!')

# Generate SI Table 4 (training on "UCD")
expName = 'EXP_Table_SI_4_training_on_UCD'
numOfSeeds = 1

subject_indices_Dreams = np.arange(1, len(v120Dreams) + 1)
subject_indices_Tra = subject_normalization(v120T[:, :-1], Original_subject_indices_Tra_without_wake)
subject_indices_Val = subject_normalization(v120V[:, :-1], Original_subject_indices_Val_without_wake)
subject_indices_UCD = subject_normalization(v120UCD[:, :-1], Original_subject_indices_UCD_without_wake)

# Training SVM model
# ...

print(f'{expName} is Done!')

def get_preprocess_pdfeat(pdfeat):
    n, m = pdfeat.shape
    pmat = pdfeat.copy()
    for i in range(m - 1):
        pmat = pmat[~np.isnan(pmat[:, i])]
        pmat = pmat[pmat[:, i] < 1e8]
    return pmat

# Clear workspace
v90T, v90V, v90Dreams, v90UCD, v120T, v120V, v120Dreams, v120UCD = None, None, None, None, None, None, None, None

# Reload features
v90T = ftH90NewPS2_CGMH[:, :16]
v90V = ftH90NewPS2_Val[:, :16]
v90Dreams = ftH90NewPS2_DREAM[:, :16]
v90UCD = ftH90NewPS2_UCD[:, :16]

v120T = np.concatenate((ftR120NewPS2_CGMH[:, :16], ftR120NewPS2_CGMH[:, 24:40]), axis=1)
v120V = np.concatenate((ftR120NewPS2_Val[:, :16], ftR120NewPS2_Val[:, 24:40]), axis=1)
v120Dreams = np.concatenate((ftR120NewPS2_DREAM[:, :16], ftR120NewPS2_DREAM[:, 24:40]), axis=1)
v120UCD = np.concatenate((ftR120NewPS2_UCD[:, :16], ftR120NewPS2_UCD[:, 24:40]), axis=1)

# Filter data
Original_subject_indices_Tra_without_wake = get_preprocess_pdfeat(v120T)
Original_subject_indices_Val_without_wake = get_preprocess_pdfeat(v120V)
Original_subject_indices_Dream_without_wake = get_preprocess_pdfeat(v120Dreams)
Original_subject_indices_UCD_without_wake = get_preprocess_pdfeat(v120UCD)

def statistics_in_avg(collection):
    return np.mean(collection)

def subject_get_model(data, target, model_type):
    # Perform classification
    # ...

    # Compute performance measures
    # ...

    # Return performance measures
    # ...

# Example usage
data = v90T[:, :-1]
target = v90T[:, -1]
model_type = 'svm'

performance_measures = subject_get_model(data, target, model_type)
print(performance_measures)
