import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import streamlit as st

# PHASE 1
# Data acquisition and exploration.
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

df = df.drop(columns=['Unnamed: 0']) # To eliminate the first column showing row indexes. 
print(df.describe()) # For some descriptive statistics.
""" 
          carat         depth         table         price             x             y             z
count  53940.000000  53940.000000  53940.000000  53940.000000  53940.000000  53940.000000  53940.000000
mean       0.797940     61.749405     57.457184   3932.799722      5.731157      5.734526      3.538734
std        0.474011      1.432621      2.234491   3989.439738      1.121761      1.142135      0.705699
min        0.200000     43.000000     43.000000    326.000000      0.000000      0.000000      0.000000
25%        0.400000     61.000000     56.000000    950.000000      4.710000      4.720000      2.910000
50%        0.700000     61.800000     57.000000   2401.000000      5.700000      5.710000      3.530000
75%        1.040000     62.500000     59.000000   5324.250000      6.540000      6.540000      4.040000
max        5.010000     79.000000     95.000000  18823.000000     10.740000     58.900000     31.800000 
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



# PHASE 2
# Feature engineering and pre-processing. 

# 2. a) Let's start with feature selection using a correlation heat map.

# Encode categorical columns i.e.
""" 
cut            object
color          object
clarity        object
"""
df_encoded = df.copy()
for col in ['cut', 'color', 'clarity']:
     df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Compute correlation matrix
corr = df_encoded.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap - Diamond Dataset')
plt.tight_layout()
# plt.show()

""" 
Most Important Features to Include in Your Model:
These are features with strong correlation to price (usually above 0.7 is considered strong):

1) carat

2) x, y, z (note: these are strongly correlated with carat, so might be redundant — more on that below)

High risk of Multicollinearity 
Check this:

carat vs x: 0.98

x vs y: 0.97

x vs z: 0.97

etc...

These values show multi-collinearity — when features are too strongly correlated with each other. That’s bad for linear regression because:

It causes instability in coefficient estimates.

The model may "double-count" similar information.

Solution:
Use only one or two of them:

Either use just carat (it captures size best), OR

Use one dimension like x instead of all three.

In Summary:
BEST FEATURES FOR LINEAR REGRESSION:
carat (top predictor)

Optionally: one of x, y, or z — but not all

Maybe color — weak, but may help

Drop: depth, cut, clarity, table — not helpful or redundant

For this case, I will use "CARAT", "Z", and "COLOR"
"""
# We now generate a feature importance barplot to further check the outcome of the heat map results.
# Compute correlation matrix
corr_matrix = df_encoded.corr()

# Get correlation with 'price' and sort
price_corr = corr_matrix['price'].drop('price')  # Remove self-correlation (1.0)
price_corr_sorted = price_corr.sort_values(ascending=True)  # Ascending for horizontal bars

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=price_corr_sorted, y=price_corr_sorted.index, palette='coolwarm')
plt.title('Feature Correlation with Diamond Price')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
plt.show() # From the results, we will be using carat, z, and color.




















































