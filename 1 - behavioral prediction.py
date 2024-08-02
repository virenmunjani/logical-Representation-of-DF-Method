from sklearn.svm import SVC

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(behavioral_dataset_features, behavioral_dataset_labels)

behavioral_predictions = svm_classifier.predict(test_data)

def h_function(x):
    decision = svm_classifier.decision_function(x)
    return decision
