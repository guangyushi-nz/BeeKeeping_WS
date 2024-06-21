import numpy as np
import librosa
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def process_audio_files(queen_training_file, no_queen_training_file, queen_validation_file, no_queen_validation_file):
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

    print("Number of Queen Segments:", len(queen_training_data_segments))
    print("Number of No Queen Segments:", len(no_queen_training_data_segments))

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

    X = np.vstack((queen_mfccs, no_queen_mfccs))
    y = np.hstack((queen_labels, no_queen_labels))

    X = X.reshape(X.shape[0], -1) 

     # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an SVM classifier (you can choose different kernels and parameters)
    clf = svm.SVC(kernel='linear')

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Calculate accuracy and print the classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of MFCC Features: {accuracy * 100:.2f}%")

    report = classification_report(y_test, y_pred)
    print("\nClassification Report:\n", report)


    # Train the classifier
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Load and process validation data
    queen_validation_data, queen_validation_sampling_rate = librosa.load(queen_validation_file, sr=None)
    no_queen_validation_data, no_queen_validation_sampling_rate = librosa.load(no_queen_validation_file, sr=None)

    queen_validation_data_segments = [
        queen_validation_data[i:i + segment_length_frames] 
        for i in range(0, len(queen_validation_data), segment_length_frames)
    ]

    no_queen_validation_data_segments =  [
        no_queen_validation_data[i:i + segment_length_frames] 
        for i in range(0, len(no_queen_validation_data), segment_length_frames)
    ]

    print("Number of Queen Segments:", len(queen_validation_data_segments))
    print("Number of No Queen Segments:", len(no_queen_validation_data_segments))

    max_length = max(len(segment) for segment in queen_validation_data_segments)
    queen_validation_data_segments_padded = [
        np.pad(segment, (0, max_length - len(segment)))
        for segment in queen_validation_data_segments
    ]

    max_length = max(len(segment) for segment in no_queen_validation_data_segments)
    no_queen_validation_data_segments_padded = [
        np.pad(segment, (0, max_length - len(segment)))
        for segment in no_queen_validation_data_segments
    ]

    queen_validation_mfccs = []  
    queen_validation_labels = []
    no_queen_validation_mfccs = [] 
    no_queen_validation_labels = []

    target_number_of_columns = min_length

    for segment in queen_validation_data_segments_padded:
        queen_mfcc = librosa.feature.mfcc(y=segment, sr=queen_sampling_rate, n_mfcc=13)
        queen_validation_mfcc_resampled = queen_mfcc[:, :target_number_of_columns]
        queen_validation_mfccs.append(queen_validation_mfcc_resampled)
        queen_validation_labels.append(1)  
    
    for segment in no_queen_validation_data_segments_padded:
        no_queen_mfcc = librosa.feature.mfcc(y=segment, sr=queen_sampling_rate, n_mfcc=13)
        no_queen_validation_mfcc_resampled = no_queen_mfcc[:, :target_number_of_columns]
        no_queen_validation_mfccs.append(no_queen_validation_mfcc_resampled)
        no_queen_validation_labels.append(0)  

    X_validation = np.vstack((queen_validation_mfccs, no_queen_validation_mfccs))
    y_validation = np.hstack((queen_validation_labels, no_queen_validation_labels))

    X_validation = X_validation.reshape(X_validation.shape[0], -1) 

    # Make predictions on the validation data
    y_pred_validation = clf.predict(X_validation)

    # Calculate accuracy and print the classification report for validation data
    accuracy_validation = accuracy_score(y_validation, y_pred_validation)
    print(f"Accuracy of MFCC Features on Validation Data: {accuracy_validation * 100:.2f}%")

    report_validation = classification_report(y_validation, y_pred_validation)
    print("\nClassification Report on Validation Data:\n", report_validation)
