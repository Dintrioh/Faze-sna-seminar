import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def subject_normalization(data, indices):
    normalized_data = data[indices, :]
    return normalized_data

def run_ceeds(exp_name, num_of_ceeds, pdfeat_train_w_label, pdfeat_val_w_label, pdfeat_dreams_w_label, pdfeat_ucd_w_label, id_of_training_set, model_mode):
    # Step 1: Final statistics
    collection_tra = []
    collection_val = []
    collection_dreams = []
    collection_ucd = []

    # Step 2: Ceed Iterations
    for ceed in range(1, num_of_ceeds + 1):
        result_tra = get_model(f"{exp_name}_Training", ceed, pdfeat_train_w_label, pdfeat_train_w_label, id_of_training_set, 1, model_mode)
        collection_tra.append(result_tra)

        result_val = get_model(f"{exp_name}_Validation", ceed, pdfeat_train_w_label, pdfeat_val_w_label, id_of_training_set, 1, model_mode)
        collection_val.append(result_val)

        result_dreams = get_model(f"{exp_name}_Dreams", ceed, pdfeat_train_w_label, pdfeat_dreams_w_label, id_of_training_set, 2, model_mode)
        collection_dreams.append(result_dreams)

        result_ucd = get_model(f"{exp_name}_UCD", ceed, pdfeat_train_w_label, pdfeat_ucd_w_label, id_of_training_set, 3, model_mode)
        collection_ucd.append(result_ucd)

    # Step 3: Save results
    np.save(f"./Table_Infos/{exp_name}_Training.npy", np.array(collection_tra))
    np.save(f"./Table_Infos/{exp_name}_Validation.npy", np.array(collection_val))
    np.save(f"./Table_Infos/{exp_name}_Dreams.npy", np.array(collection_dreams))
    np.save(f"./Table_Infos/{exp_name}_UCD.npy", np.array(collection_ucd))

    # Step 4: Save average statistics
    result_tra_avg = np.mean(collection_tra, axis=0)
    result_val_avg = np.mean(collection_val, axis=0)
    result_dreams_avg = np.mean(collection_dreams, axis=0)
    result_ucd_avg = np.mean(collection_ucd, axis=0)

    np.save(f"./Table_Infos/{exp_name}_Training_avg.npy", result_tra_avg)
    np.save(f"./Table_Infos/{exp_name}_Validation_avg.npy", result_val_avg)
    np.save(f"./Table_Infos/{exp_name}_Dreams_avg.npy", result_dreams_avg)
    np.save(f"./Table_Infos/{exp_name}_UCD_avg.npy", result_ucd_avg)

def get_model(exp_name, ceed, pdfeat_train_w_label, pdfeat_test_w_label, id_of_training_set, id_of_test_set, model_mode):
    # Perform model training and testing
    train_data = pdfeat_train_w_label[:, :-1]
    train_labels = pdfeat_train_w_label[:, -1]
    test_data = pdfeat_test_w_label[:, :-1]
    test_labels = pdfeat_test_w_label[:, -1]

    if model_mode == 'SVM':
        model = SVC()
    elif model_mode == 'RT':
        model = RandomForestClassifier()

    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)

    # Compute evaluation metrics
    TP = np.sum((predictions == 1) & (test_labels == 1))
    TN = np.sum((predictions == 0) & (test_labels == 0))
    FP = np.sum((predictions == 1) & (test_labels == 0))
    FN = np.sum((predictions == 0) & (test_labels == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    result = [exp_name, ceed, id_of_training_set, id_of_test_set, accuracy, sensitivity, specificity, precision, f1_score]
    return result

# Load data and files
pdfeat_train_w_label = np.load("./data/pdfeat_train_w_label.npy")
pdfeat_val_w_label = np.load("./data/pdfeat_val_w_label.npy")
pdfeat_dreams_w_label = np.load("./data/pdfeat_dreams_w_label.npy")
pdfeat_ucd_w_label = np.load("./data/pdfeat_ucd_w_label.npy")

# Perform subject normalization
indices_cgmh = np.arange(0, pdfeat_train_w_label.shape[0]) # Replace with appropriate indices
indices_dreams = np.arange(0, pdfeat_dreams_w_label.shape[0]) # Replace with appropriate indices
indices_ucd = np.arange(0, pdfeat_ucd_w_label.shape[0]) # Replace with appropriate indices

normalized_cgmh = subject_normalization(pdfeat_train_w_label, indices_cgmh)
normalized_dreams = subject_normalization(pdfeat_dreams_w_label, indices_dreams)
normalized_ucd = subject_normalization(pdfeat_ucd_w_label, indices_ucd)

# Define parameters
exp_name = "Experiment1"
num_of_ceeds = 10
id_of_training_set = 1
model_mode = "SVM"

# Run CEEDS for different datasets
run_ceeds(exp_name, num_of_ceeds, normalized_cgmh, pdfeat_val_w_label, normalized_dreams, normalized_ucd, id_of_training_set, model_mode)
