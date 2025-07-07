import pandas as pd 
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

st.set_page_config(page_title="💎 Diamond Price Estimator", page_icon="💰", 
                   layout="centered")

# Sidebar: branding
st.sidebar.image("https://i.imgur.com/ExdKOOz.png", width=200)
st.sidebar.title("Kamusi Data Lab")
st.sidebar.markdown("Predict your diamond's price using Machine Learning 💻")

# Main Title
st.title("💎 Diamond Price Prediction App")
st.markdown("""
<style>
.big-font {
    font-size:20px;
    font-weight:600;
    color:#2f4f4f;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Enter your diamond\'s features below:</p>', unsafe_allow_html=True)

# Layout input columns
col1, col2 = st.columns(2)

with col1:
    carat = st.slider("Carat", 0.2, 5.0, 0.5, 0.01)
    depth = st.slider("Depth %", 50.0, 70.0, 61.0, 0.1)
    cut = st.selectbox("Cut", ["Ideal", "Premium", "Very Good", "Good", "Fair"])
    clarity = st.selectbox("Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])

with col2:
    table = st.slider("Table %", 50.0, 70.0, 57.0, 0.1)
    x = st.slider("Length (mm)", 3.0, 10.0, 5.0, 0.1)
    y = st.slider("Width (mm)", 3.0, 10.0, 5.0, 0.1)
    z = st.slider("Height (mm)", 2.0, 6.0, 3.0, 0.1)
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])

# Convert input to model-ready format
input_dict = {
    'carat': carat,
    'depth': depth,
    'table': table,
    'x': x,
    'y': y,
    'z': z,
    'cut_Premium': int(cut == 'Premium'),
    'cut_Very Good': int(cut == 'Very Good'),
    'cut_Good': int(cut == 'Good'),
    'cut_Fair': int(cut == 'Fair'),
    'color_E': int(color == 'E'),
    'color_F': int(color == 'F'),
    'color_G': int(color == 'G'),
    'color_H': int(color == 'H'),
    'color_I': int(color == 'I'),
    'color_J': int(color == 'J'),
    'clarity_VVS1': int(clarity == 'VVS1'),
    'clarity_VVS2': int(clarity == 'VVS2'),
    'clarity_VS1': int(clarity == 'VS1'),
    'clarity_VS2': int(clarity == 'VS2'),
    'clarity_SI1': int(clarity == 'SI1'),
    'clarity_SI2': int(clarity == 'SI2'),
    'clarity_I1': int(clarity == 'I1')
}

input_df = pd.DataFrame([input_dict])

st.info("✨ Model not yet connected. This is just a front end preview")


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

