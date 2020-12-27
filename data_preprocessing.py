"""
Created on Thu Dec 24 23:15:33 2020

@author: mridul
"""

# Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Reading CSV File
data = pd.read_csv(r"C:\\Users\\Mridul\\Desktop\\bitstampUSD.csv")

# Checkin first and last few records of the dataset.
data.head()
data.tail()

# Checking if null values are present.
data.isnull().sum()

# Checking number of records.
data.shape

# Dropping rows with nan value
data = data.dropna()

# No null values present.
data.isnull().sum()

# Remaining dataset size
data.shape

# Checkin first few records of the dataset.
data.head()

# As the index of rows are mismatched we use reset_index
data = data.reset_index()
data.head()


# The previous index are stored in the column which are no longer needed
del data['index']

data.to_csv("C:\\Users\\Mridul\\Desktop\\cleanbit.csv")

data.head()