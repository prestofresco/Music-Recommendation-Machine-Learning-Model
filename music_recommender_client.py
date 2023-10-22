import pandas as pd
# if this import is giving an error, open terminal and make sure that pandas is installed on your computer.
# run the command 'pip install pandas' * this also requires you have 'pip' installed (python package manager) *
# also check which version of python you are currently on with command 'python3 version'
# I am currently on python version 3.11.2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # 'pip install seaborn' if not working
from sklearn.linear_model import LogisticRegression # pip install -U scikit-learn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('Spotify_114k.csv')

# df['average'] = df.mean(axis=1, numeric_only=True)

df = df.dropna(subset=['artists', 'album_name', 'track_name']) # drop null values from the columns that contain them
df.info()
print("datatypes: \n", df.dtypes)

# print("Number of missing values in each column:")
# print(df.isnull().sum())
# print(df.describe())

# print("\nHEAD: \n", df.head(5))
# print("\nTAIL: \n", df.tail(5))

# print(df.head().transpose())

le = LabelEncoder()
model = LogisticRegression(solver='liblinear', max_iter=1000)

le.fit_transform(df)

# Split the dataset into features and target
target = df['track_id']
features = df.drop('track_id', axis=1)


# Split the dataset into training and testing sets
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=7)

# Split the dataset into training and testing sets
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=7)

# Print the sizes of the training and testing sets
print('Training Sets: Count of Feature instances: ', len(feature_train))
print('Training Sets: Count of Target instances: ', len(target_train))

print('Testing Sets: Count of Feature instances: ', len(feature_test))
print('Testing Sets: Count of Target instances: ', len(target_test))

# Print the first few rows of the features and target dataframes
print('\nFeatures:')
print(features.head())

print('\nTarget:')
print(target.head())


# Train the model on the training data
model.fit(feature_train, target_train.values.ravel())

# Predict the targets for the given feature_test
target_predicted = model.predict(feature_test)


# Measuring the performance of the model
model_score = model.score(feature_test, target_test)
print("Model Score: ", model_score)

accuracy = metrics.accuracy_score(target_test, target_predicted)
print("Accuracy Score: ", accuracy)

accuracy = metrics.precision_score(target_test, target_predicted, average='weighted')
print("Precision Score: ", accuracy)

accuracy = metrics.recall_score(target_test, target_predicted , average='weighted')
print("Recall Score: ", accuracy)
