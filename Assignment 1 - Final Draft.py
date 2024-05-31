#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Download the provided dataset and save it in your working directory.

data = r"C:\Users\marge\Downloads\Weekly_Assignment (1).xlsx"
xls = pd.ExcelFile(data)



# Display the sheet names to understand the structure of the file
sheet_names = xls.sheet_names
sheet_names

df = pd.read_excel(data, sheet_name='Weekly_Assignment')



# In[1]:


#Import the necessary libraries, such as pandas and numpy.
import numpy as np
import pandas as pd


# In[3]:


# Load the dataset into a Pandas DataFrame
df = pd.read_excel(data)

# Display the DataFrame
print(df)
       
       
       
       


# In[12]:


#Identify the columns with missing values using appropriate Pandas methods.

print(df.isnull().sum())


# In[31]:


# Use the fillna() method to handle missing values with the mean for numerical columns and mode for non numeri

# Fill missing values
df_filled = df.copy()
for column in df_filled.columns:
    if pd.api.types.is_numeric_dtype(df_filled[column]):
        df_filled[column].fillna(df_filled[column].mean(), inplace=True)
    else:
        df_filled[column].fillna(df_filled[column].mode()[0], inplace=True)

# Display the DataFrame with filled missing values
print(df_filled)


# In[41]:


#create a new dataset with the modified dataset. That is, save the dataset after handling missing values

filled_file_path = r"C:\Users\marge\Downloads\Weekly_Assignment_Filled.xlsx"
df_filled.to_excel(filled_file_path, index=False)
print("Dataset after handling missing values saved to:", filled_file_path)





# In[52]:


#Handle outliers

# Load the dataset (assuming missing values have already been handled)
file_path = r"C:\Users\marge\Downloads\Weekly_Assignment_Filled.xlsx"
df = pd.read_excel(file_path)

# Function to detect outliers using z-scores
def detect_outliers_zscores(data, threshold=2):
    mean = np.mean(data)
    std = np.std(data)
    outliers = data[np.abs((data - mean) / std) > threshold]
    return outliers

# Detect and print outliers for all numerical columns
for column in df.select_dtypes(include=[np.number]).columns:
    data = df[column].dropna()
    outliers = detect_outliers_zscores(data)
    print(f"Outliers from z-score method in '{column}': {outliers.values}")



# In[53]:


# Remove outliers from the dataset
# I decided to remove the outliers from the dataset so that they do not distort the final outcome of the analysis. 
# I recognise that this method reduces the number of variables within the dataset and accept that risk.

for column in df.select_dtypes(include=[np.number]).columns:
    data = df[column].dropna()
    outliers = detect_outliers_zscores(data)
    df = df[~df[column].isin(outliers)]


# Save the cleaned dataset to a new file
cleaned_file_path = r"C:\Users\marge\Downloads\Weekly_Assignment_Cleaned.xlsx"
df.to_excel(cleaned_file_path, index=False)
print("Cleaned dataset saved to:", cleaned_file_path)


# Visualize the outliers using box plots for all numerical columns
plt.figure(figsize=(15, 10))
df.select_dtypes(include=[np.number]).boxplot()
plt.title('Box plots of all numerical columns after outlier removal')
plt.show()

