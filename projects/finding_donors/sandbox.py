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

categorical  = ['workclass','education_level','marital-status','occupation','relationship','race','sex','native-country']
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

beta_sqrd = 0.5**2
fscore = (1.0 + beta_sqrd) * precision / ((beta_sqrd * precision) + 1.0)

# Print the results
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)