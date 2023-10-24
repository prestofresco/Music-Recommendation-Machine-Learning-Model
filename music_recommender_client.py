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


df = pd.read_csv('Spotify_114k.csv', index_col=0) # specify to ignore the unnamed first column

# df['average'] = df.mean(axis=1, numeric_only=True)

df = df.dropna(subset=['artists', 'album_name', 'track_name']) # drop null values from the columns that contain them
df.info()
print("datatypes: \n", df.dtypes)

# print("Number of missing values in each column:")
# print(df.isnull().sum())
# print(df.describe())

# print("\nHEAD: \n", df.head(5))
# print("\nTAIL: \n", df.tail(5))
# print("unique track names:", df['track_name'].nunique())
# print("shape of dataset:", df.shape)

# print(df.head().transpose())

le = LabelEncoder()
model = LogisticRegression(solver='liblinear', max_iter=1000)

# Encode the string values
for label, content in df.items():
    if content.dtype == 'object':
        df[label] = le.fit_transform(df[label])

print("datatypes: \n", df.dtypes, "\n\n")



# ----------------------------------------------------------------------

# Displaying the correlation data
# Calculate the correlation between danceability and all input variables
corr_matrix = df.corr()["danceability"].drop("danceability")

# Create a bar plot using seaborn
sns.set(style="white")
plt.figure(figsize=(12,6))
ax = sns.barplot(x=corr_matrix.index, y=corr_matrix.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.title("Correlation between Danceability and Input Variables")
plt.show()


# # Split the dataset into features and target
# target = df['track_genre']
# features = df.drop('track_genre', axis=1)


# # Split the dataset into training and testing sets
# feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=7)

# # Split the dataset into training and testing sets
# feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=7)

# # Print the sizes of the training and testing sets
# print('Training Sets: Count of Feature instances: ', len(feature_train))
# print('Training Sets: Count of Target instances: ', len(target_train))

# print('Testing Sets: Count of Feature instances: ', len(feature_test))
# print('Testing Sets: Count of Target instances: ', len(target_test))

# # Print the first few rows of the features and target dataframes
# print('\nFeatures:')
# print(features.head())

# print('\nTarget:')
# print(target.head())


# # Train the model on the training data
# model.fit(feature_train, target_train.values.ravel())

# # Predict the targets for the given feature_test
# target_predicted = model.predict(feature_test)


# # Measuring the performance of the model
# model_score = model.score(feature_test, target_test)
# print("Model Score: ", model_score)

# accuracy = metrics.accuracy_score(target_test, target_predicted)
# print("Accuracy Score: ", accuracy)

# accuracy = metrics.precision_score(target_test, target_predicted, average='weighted')
# print("Precision Score: ", accuracy)

# accuracy = metrics.recall_score(target_test, target_predicted , average='weighted')
# print("Recall Score: ", accuracy)
