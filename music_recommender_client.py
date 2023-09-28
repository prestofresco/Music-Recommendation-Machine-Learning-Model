import pandas as pd
# if this import is giving an error, open terminal and make sure that pandas is installed on your computer.
# run the command 'pip install pandas'
# also check which version of python you are currently on with command 'python3 version'
# I am currently on python version 3.11.2

df = pd.read_csv('Spotify_114k.csv')
df.info()

print("\nHEAD: \n", df.head(5))
print("\nTAIL: \n", df.tail(5))

print(df.describe())