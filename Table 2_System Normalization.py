import numpy as np

def subject_normalization(data, annotations):
    normalized_data = []
    
    for i, subject_data in enumerate(data):
        subject_annotations = annotations[i]
        awake_indices = np.where(subject_annotations == 'Awake')[0]
        
        if len(awake_indices) == 0:
            # No awake states in the subject's annotations, normalize by subtracting mean
            mean = np.mean(subject_data)
            normalized_subject_data = subject_data - mean
        else:
            # Normalize by subtracting mean of awake states
            awake_data = subject_data[awake_indices]
            mean = np.mean(awake_data)
            normalized_subject_data = subject_data - mean
        
        normalized_data.append(normalized_subject_data)
    
    return normalized_data

# Example usage
data = [
    np.array([1, 2, 3, 4, 5]),
    np.array([6, 7, 8, 9, 10]),
    np.array([11, 12, 13, 14, 15])
]

annotations = [
    np.array(['Awake', 'Awake', 'Awake', 'Awake', 'Awake']),
    np.array(['Non-REM', 'Non-REM', 'Non-REM', 'Non-REM', 'Non-REM']),
    np.array(['REM', 'REM', 'REM', 'REM', 'REM'])
]

normalized_data = subject_normalization(data, annotations)
for subject_data in normalized_data:
    print(subject_data)
