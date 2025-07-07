import pandas as pd 
import streamlit as st

# 1. a) Load the dataset.
data = r"C:\Users\patri\Downloads\diamonds.csv.zip" # I've used an absolute path. Use relative path to access the data.
df = pd.read_csv(data) # Convert csv into dataframe using pandas.


print(df.head(10)) # The data looks good
print(df.shape) # (53940 rows, 11 columns) Good data to work with
print(df.columns) #['Unnamed: 0', 'carat', 'cut', 'color', 'clarity', 'depth', 'table','price', 'x', 'y', 'z']
data_types = df.dtypes
print(data_types) # Investigating the data types in the dataset
"""
Unnamed: 0      int64
carat         float64
cut            object
color          object
clarity        object
depth         float64
table         float64
price           int64
x             float64
y             float64
z             float64
"""

# 1. b) Explanatory data analysis.

print(df.isnull().sum().sort_values(ascending = False))
# This will show me the number of empty cells per row with
# empty columns such that I can find out with columns to let go
# depending on the percentage number of empty cells w.r.t the entire dataset.

"""
dtype: object
Unnamed: 0    0
carat         0
cut           0
color         0
clarity       0
depth         0
table         0
price         0
x             0
y             0
z             0
     """
# This indicates that the data we are working with is of very high quality.

# Just for the sake, I'll use a for oop to iterate over the dataset columns
# The purpose of this will be to determine the percentage number of empty cells w.r.t the entire dataset.

for column in df.columns:
     percentage_empty = df[column].isnull().mean()
     print(column + " ---> " + str(percentage_empty)+ " '%' empty cells.")

""" 
Unnamed: 0 ---> 0.0 % empty cells.
carat ---> 0.0 % empty cells.
cut ---> 0.0 % empty cells.
color ---> 0.0 % empty cells.
clarity ---> 0.0 % empty cells.
depth ---> 0.0 % empty cells.
table ---> 0.0 % empty cells.
price ---> 0.0 % empty cells.
x ---> 0.0 % empty cells.
y ---> 0.0 % empty cells.
z ---> 0.0 % empty cells.
"""     

# Since we don't have missing data to deal with, 
# There won't be any need to eliminate rows or columns for that reason, for now.

# In the event that we had missing values and the percentage was high,
# Would then have to eliminate the given columns,
# But if it wasn't then we would simply use the .fillna() method and the .join() method to fill the empty cells with the mode/median/mean of the given column.





















































