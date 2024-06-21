import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import IPython.display as ipd
import librosa
import soundfile as sf
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def process_audio_data(queen_file, no_queen_file):
    def load_and_process_data(training_file, label):
        training_data, sampling_rate = librosa.load(training_file, sr=None)

        # Create segments for training data
        segment_length = 3  # seconds
        segment_length_frames = int(segment_length * sampling_rate)
        training_data_segments = [
            training_data[i:i + segment_length_frames]
            for i in range(0, len(training_data), segment_length_frames)
        ]

        # Extract MFCC features and assign labels
        mfccs = []
        labels = []

        for segment in training_data_segments:
            mfcc = librosa.feature.mfcc(y=segment, sr=sampling_rate, n_mfcc=13)
            mfccs.append(mfcc)
            labels.append(label)

        return mfccs, labels

    # Load and process data for queen class
    queen_mfccs, queen_labels = load_and_process_data(queen_file, label=1)

    # Load and process data for no queen class
    no_queen_mfccs, no_queen_labels = load_and_process_data(no_queen_file, label=0)

    # Calculate the minimum length among all segments
    min_length = min(mfcc.shape[1] for mfcc in queen_mfccs + no_queen_mfccs)

    # Trim MFCC matrices to the minimum length
    queen_mfccs = [mfcc[:, :min_length] for mfcc in queen_mfccs]
    no_queen_mfccs = [mfcc[:, :min_length] for mfcc in no_queen_mfccs]

    # Combine the data from "queen" and "no queen" segments and labels:
    X = np.vstack((queen_mfccs, no_queen_mfccs))
    y = np.hstack((queen_labels, no_queen_labels))

    # Reshape X to have two dimensions
    X = X.reshape(X.shape[0], -1)  # Flatten the last two dimensions

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

    # Check the number of columns in each array
    queen_numberOfColumn = set(mfcc.shape[1] for mfcc in queen_mfccs)
    no_queen_numberOfColumn = set(mfcc.shape[1] for mfcc in no_queen_mfccs)

    queen_validation_file = "QueenBee_Testing_15mins.wav"
    no_queen_validation_file = "No_QueenBee_Testing_15mins.wav"

    queen_validation_data, queen_validation_sampling_rate = librosa.load(queen_validation_file, sr=None)
    no_queen_validation_data, no_queen_validation_sampling_rate = librosa.load(no_queen_validation_file, sr=None)

    segment_length = 3  # seconds

    segment_length_frames = int(segment_length * queen_sampling_rate)

    # Create segments for queen and no queen training data
    segment_length = 3  # seconds
    queen_validation_data_segments = [
                                    queen_validation_data[i:i + segment_length_frames] 
                                        for i in range(0, len(queen_validation_data), segment_length_frames)
                                    ]

    no_queen_validation_data_segments =  [
                                    no_queen_validation_data[i:i + segment_length_frames] 
                                        for i in range(0, len(no_queen_validation_data), segment_length_frames)
                                        ]

    # Ensure you have 900 segments for each class
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

    # Use the maximum number of frames as the target for validation set
    print("Testing", queen_numberOfColumn)
    target_number_of_columns = target_number_of_columns = list(queen_numberOfColumn)[0]

    # Assign labels for Queen class
    for segment in queen_validation_data_segments_padded:
        queen_mfcc = librosa.feature.mfcc(y=segment, sr=queen_sampling_rate, n_mfcc=13)
        queen_validation_mfcc_resampled = queen_mfcc[:, :target_number_of_columns]
        queen_validation_mfccs.append(queen_validation_mfcc_resampled)
        queen_validation_labels.append(1)  # Assign label 1 for Queen class
        
    # Assign labels for No Queen class
    for segment in no_queen_validation_data_segments_padded:
        no_queen_mfcc = librosa.feature.mfcc(y=segment, sr=queen_sampling_rate, n_mfcc=13)
        no_queen_validation_mfcc_resampled = no_queen_mfcc[:, :target_number_of_columns]
        no_queen_validation_mfccs.append(no_queen_validation_mfcc_resampled)
        no_queen_validation_labels.append(0)  # Assign label 0 for No Queen class

    # Check the number of columns in each array
    queen_mfcc_numberOfColumn = set(mfcc.shape[1] for mfcc in queen_validation_mfccs)
    no_queen_mfcc_numberOfColumn = set(mfcc.shape[1] for mfcc in no_queen_validation_mfccs)

    print("Queen MFCC Number Of Column:", queen_mfcc_numberOfColumn)
    print("No Queen MFCC Number Of Column:", no_queen_mfcc_numberOfColumn)

    # Combine the data from "queen" and "no queen" segments and labels:
    X_validation = np.vstack((queen_validation_mfccs, no_queen_validation_mfccs))
    y_validation = np.hstack((queen_validation_labels, no_queen_validation_labels))

    # Reshape X to have two dimensions
    X_validation = X_validation.reshape(X_validation.shape[0], -1)  # Flatten the last two dimensions

    # Make predictions on the test data
    y_pred = clf.predict(X_validation)

    # Calculate accuracy and print the classification report
    accuracy = accuracy_score(y_validation, y_pred)
    print(f"Accuracy of MFCC Features: {accuracy * 100:.2f}%")

    report = classification_report(y_validation, y_pred)
    print("\nClassification Report:\n", report)
