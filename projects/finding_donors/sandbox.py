#############################################################

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
#from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
#display(data.head(n=1))

#############################################################
income_counts = data["income"].value_counts()

n_records = data.shape[0]
n_greater_50k = income_counts[">50K"]
n_at_most_50k = income_counts["<=50K"]
greater_percent = 100.0 * n_greater_50k / n_records

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

#############################################################

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
#vs.distribution(data)

#############################################################

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
#vs.distribution(features_log_transformed, transformed = True)

#############################################################

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
#display(features_log_minmax_transform.head(n = 5))

############################################################

categorical = [
    'workclass',
    'education_level',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country'
]

features_final = pd.get_dummies(data = features_log_minmax_transform, columns = categorical)

income = income_raw.apply(lambda x: int(x==">50K"))
encoded = list(features_final.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
#print encoded

############################################################

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

############################################################

true_positives = np.sum(income.values)
false_positives = income.shape[0] - true_positives
accuracy = true_positives / float(income.shape[0])
recall = 1.0
precision = accuracy
beta = 0.5
beta_sqrd = beta*beta
fscore = (1.0 + beta_sqrd) * precision / ((beta_sqrd * precision) + 1.0)

# Print the results
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)

############################################################

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test, n_train_predictions = 300, beta = 0.5):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    start = time()
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    results['train_time'] = end - start

    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:n_train_predictions])
    end = time()
    results['pred_time'] = end - start

    results['acc_train'] = accuracy_score(y_train[:len(predictions_train)], predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    results['f_train'] = fbeta_score(y_train[:len(predictions_train)], predictions_train, beta = beta)
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = beta)

    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)

    return results

############################################################
#from sklearn.svm import SVC
#from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

random_seed = 0xDEADBEEF
"""
clf_A = LinearSVC(random_state=random_seed)
clf_B = KNeighborsClassifier()
clf_C = DecisionTreeClassifier(random_state=random_seed)

samples_100 = len(y_train)
samples_10 = samples_100 / 10
samples_1 = samples_100 / 100

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)
"""

############################################################

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

clf = LinearSVC(random_state=random_seed)

parameters = {
    'C': [1e-2,1,100,1e5],
    'tol': [1e-4,1e-2,1],
    'class_weight': [None, 'balanced']
}

scorer = make_scorer(fbeta_score, beta = beta)
grid_obj = GridSearchCV(clf,parameters,scorer,verbose = 2, n_jobs = 2)

grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))

############################################################
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(random_state=random_seed)
model.fit(X_train, y_train)
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)