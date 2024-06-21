import numpy as np
import librosa
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


def process_audio_files(queen_training_file, no_queen_training_file, queen_testing_file, no_queen_testing_file):
    # Load and process training data
    queen_training_data, queen_sampling_rate = librosa.load(queen_training_file, sr=None)
    no_queen_training_data, no_queen_sampling_rate = librosa.load(no_queen_training_file, sr=None)

    segment_length = 3  # seconds
    segment_length_frames = int(segment_length * queen_sampling_rate)

    queen_training_data_segments = [
        queen_training_data[i:i + segment_length_frames] 
        for i in range(0, len(queen_training_data), segment_length_frames)
    ]

    no_queen_training_data_segments =  [
        no_queen_training_data[i:i + segment_length_frames] 
        for i in range(0, len(no_queen_training_data), segment_length_frames)
    ]

    print("Number of Queen Training Segments:", len(queen_training_data_segments))
    print("Number of No Queen Training Segments:", len(no_queen_training_data_segments))

    queen_mfccs = []  
    queen_labels = []
    no_queen_mfccs = [] 
    no_queen_labels = []

    for segment in queen_training_data_segments:
        queen_mfcc = librosa.feature.mfcc(y=segment, sr=queen_sampling_rate, n_mfcc=13)
        queen_mfccs.append(queen_mfcc)
        queen_labels.append(1)  

    for segment in no_queen_training_data_segments:
        no_queen_mfcc = librosa.feature.mfcc(y=segment, sr=queen_sampling_rate, n_mfcc=13)
        no_queen_mfccs.append(no_queen_mfcc)
        no_queen_labels.append(0)  

    min_length = min(mfcc.shape[1] for mfcc in queen_mfccs + no_queen_mfccs)

    queen_mfccs = [mfcc[:, :min_length] for mfcc in queen_mfccs]
    no_queen_mfccs = [mfcc[:, :min_length] for mfcc in no_queen_mfccs]

    X_train = np.vstack((queen_mfccs, no_queen_mfccs))
    y_train = np.hstack((queen_labels, no_queen_labels))

    X_train = X_train.reshape(X_train.shape[0], -1) 

    # Initialize 10-fold cross-validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


    # Train the classifier
    clf = svm.SVC(kernel='linear')

    # Lists to store performance metrics
    accuracy_scores = []

    # Loop over the folds
    for train_index, val_index in kfold.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

        # Train the classifier on the training fold
        clf.fit(X_fold_train, y_fold_train)

        # Make predictions on the validation fold
        y_pred_fold_val = clf.predict(X_fold_val)

        # Calculate accuracy and store it
        accuracy_fold_val = accuracy_score(y_fold_val, y_pred_fold_val)
        accuracy_scores.append(accuracy_fold_val)

    # Calculate and print the average accuracy across all folds
    average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    print(f"\nAverage Accuracy across 10 folds: {average_accuracy * 100:.2f}%\n")

    ######################################### Load and process testing data ##################################################
    
    queen_testing_data, queen_testing_sampling_rate = librosa.load(queen_testing_file, sr=None)
    no_queen_testing_data, no_queen_testing_sampling_rate = librosa.load(no_queen_testing_file, sr=None)

    queen_testing_data_segments = [
        queen_testing_data[i:i + segment_length_frames] 
        for i in range(0, len(queen_testing_data), segment_length_frames)
    ]

    no_queen_testing_data_segments =  [
        no_queen_testing_data[i:i + segment_length_frames] 
        for i in range(0, len(no_queen_testing_data), segment_length_frames)
    ]

    print("Number of Queen Testing Segments:", len(queen_testing_data_segments))
    print("Number of No Queen Testing Segments:", len(no_queen_testing_data_segments))

    max_length = max(len(segment) for segment in queen_testing_data_segments)
    queen_testing_data_segments_padded = [
        np.pad(segment, (0, max_length - len(segment)))
        for segment in queen_testing_data_segments
    ]

    max_length = max(len(segment) for segment in no_queen_testing_data_segments)
    no_queen_testing_data_segments_padded = [
        np.pad(segment, (0, max_length - len(segment)))
        for segment in no_queen_testing_data_segments
    ]

    queen_testing_mfccs = []  
    queen_testing_labels = []
    no_queen_testing_mfccs = [] 
    no_queen_testing_labels = []

    target_number_of_columns = min_length

    for segment in queen_testing_data_segments_padded:
        queen_mfcc = librosa.feature.mfcc(y=segment, sr=queen_sampling_rate, n_mfcc=13)
        queen_testing_mfcc_resampled = queen_mfcc[:, :target_number_of_columns]
        queen_testing_mfccs.append(queen_testing_mfcc_resampled)
        queen_testing_labels.append(1)  
    
    for segment in no_queen_testing_data_segments_padded:
        no_queen_mfcc = librosa.feature.mfcc(y=segment, sr=queen_sampling_rate, n_mfcc=13)
        no_queen_testing_mfcc_resampled = no_queen_mfcc[:, :target_number_of_columns]
        no_queen_testing_mfccs.append(no_queen_testing_mfcc_resampled)
        no_queen_testing_labels.append(0)  

    X_testing = np.vstack((queen_testing_mfccs, no_queen_testing_mfccs))
    y_testing = np.hstack((queen_testing_labels, no_queen_testing_labels))

    X_testing = X_testing.reshape(X_testing.shape[0], -1) 

    # Combine features and labels
    combined_data = np.column_stack((X_testing, y_testing))

    # Shuffle the combined data
    np.random.shuffle(combined_data)

    # Split the shuffled data back into features and labels
    X_testing_shuffled = combined_data[:, :-1]
    y_testing_shuffled = combined_data[:, -1]


    # Make predictions on the testing data
    y_pred_testing = clf.predict(X_testing_shuffled)

    # Calculate accuracy and print the classification report for testing data
    accuracy_testing = accuracy_score(y_testing_shuffled, y_pred_testing)
    print(f"\nAccuracy of MFCC Features on testing Data: {accuracy_testing * 100:.2f}%")
    
    # Print the confusion matrix
    conf_matrix = confusion_matrix(y_testing_shuffled, y_pred_testing)
    
    
    # Extract TP, TN, FP, FN from the confusion matrix
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    
    print("\nConfusion Matrix on Testing Data:\n")
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")

    report_testing = classification_report(y_testing_shuffled, y_pred_testing)
    print("\nClassification Report on testing Data:\n", report_testing)

    # Plot ROC curve
    plt.figure()
    RocCurveDisplay.from_estimator(clf, X_testing, y_testing)
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.show()

def visualize_mel_spectrogram(audio_data, sampling_rate, title):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sampling_rate)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=sampling_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()