import numpy as np
import librosa
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

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

    X_train = np.vstack((queen_mfccs, no_queen_mfccs))
    y_train = np.hstack((queen_labels, no_queen_labels))

    X_train = X_train.reshape(X_train.shape[0], -1) 

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],  # You can extend this list with other values to test
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']  # Adjust based on your requirements
    }

    # Initialize 10-fold cross-validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Train the classifier
    clf = svm.SVC(kernel='linear')
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(clf, param_grid, cv=kfold, scoring='accuracy')
    
    # Fit the model with the best parameters
    grid_search.fit(X_train, y_train)

    # Get the best SVM model
    best_model = grid_search.best_estimator_

    # Print the best parameters found
    print("Best Parameters:", grid_search.best_params_)

    # Load and process testing data
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

    print("Number of Queen Segments:", len(queen_testing_data_segments))
    print("Number of No Queen Segments:", len(no_queen_testing_data_segments))

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

    # Make predictions on the testing data
    y_pred_testing = best_model.predict(X_testing)

    # Calculate accuracy and print the classification report for testing data
    accuracy_testing = accuracy_score(y_testing, y_pred_testing)
    print(f"Accuracy of MFCC Features on testing Data: {accuracy_testing * 100:.2f}%")

    report_testing = classification_report(y_testing, y_pred_testing)
    print("\nClassification Report on testing Data:\n", report_testing)


