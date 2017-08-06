import pandas
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

classifier_used = 'randomForest'

train_data = pandas.read_csv("data/train.csv")
print('Train Data read!')
test_data = pandas.read_csv("data/test.csv")
print('Test Data read!')

# Labels of the training data
true_labels = train_data['label']

# Column names of the 784 cols
col_names = list(train_data)
col_names.remove('label')

# Extract the features
# features = train_data[features].astype(float)
features = train_data[col_names].astype(float)

# Fit the classifier
clf = RandomForestClassifier()

t0 = time()
clf.fit(features, true_labels)
tt = time()-t0
print("Classifier Fit in {} seconds".format(round(tt,3)))

# Test the Classifier's accuracy
t0 = time()
predicted_labels = clf.predict(features)
tt = time() - t0
print("Assigned labels for training_data in {} seconds".format(round(tt,3)))

accuracy_score = accuracy_score(true_labels, predicted_labels)
print("\n\nAccuracy {} %".format(round(accuracy_score*100,3)))

confusion_matrix = confusion_matrix(true_labels, predicted_labels)
print("\n\nConfusion Matrix: \n\n {}".format(confusion_matrix))

# Predict labels for the testing data
t0 = time()
predicted_labels = clf.predict(test_data)
tt = time() - t0
print("Assigned labels for testing_data in {} seconds".format(round(tt,3)))

# Write the labels to the submission_file.csv

with open('{}-submission.csv'.format(classifier_used), 'w') as f:
	f.write('ImageId,Label\n')

with open('{}-submission.csv'.format(classifier_used), 'a') as f:
	for i in range(0,len(test_data)):
		f.write('{},{}\n'.format(i+1, predicted_labels[i]))
