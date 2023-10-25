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

# ----------- data import / cleaning ----------------
df = pd.read_csv('Spotify_114k.csv', index_col=0) # specify to ignore the unnamed first column
df = df.dropna(subset=['artists', 'album_name', 'track_name']) # drop null values from the columns that contain them

# get all the columns that are type string
# unless using different dataset, result should be : ['track_id', 'artists', 'album_name', 'track_name', 'track_genre']
stringColumns = []
for label, content in df.items():
    if content.dtype == 'object':
        stringColumns.append(label)


# ----------- model declarations ---------------------
le = LabelEncoder()


# -------------------- methods -----------------------

# Encode the string values to make them useable with the ML models
# param: dataframe, returns: the new encoded dataframe
def encode(df):
    # encode the string columns
    for col in stringColumns:
        df[col] = le.fit_transform(df[col])
    return df


def decode(df):
    # decode the string columns
    for col in stringColumns:
        df[col] = le.inverse_transform(df[col])
    return df

# Displays correlation matrix data 
# Calculates the correlation between the specified column and all input variables
# param: string column name
def correlation_matrix(column):
    corr_matrix = df.corr()[column].drop(column)
    # Create a bar plot using seaborn
    sns.set(style="white")
    plt.figure(figsize=(12,6))
    ax = sns.barplot(x=corr_matrix.index, y=corr_matrix.values)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.title("Correlation between " + column + " and Input Variables")
    plt.show()


# model = LogisticRegression(solver='liblinear', max_iter=1000)

        
print("datatypes: \n", df.dtypes, "\n\n")

# Decode the string values
# for col in stringColumns:
#         df[col] = le.inverse_transform(df[col])
        
# print("datatypes after decoding: \n", df.dtypes, "\n\n")



# ---------------------------- MAIN ------------------------------------------

print('before encoding: ')
df.info()

df = encode(df)
print('after encoding: ')
df.info()

df = decode(df)
print('after decoding: ')
df.info()



# print("Number of missing values in each column:")
# print(df.isnull().sum())
# print(df.describe())







