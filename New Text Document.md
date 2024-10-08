# Diwali Sales Data Analysis

This project focuses on analyzing the Diwali Sales Data to gain insights into customer behavior and sales trends during the Diwali festival. The primary goal is to perform data processing, exploration, and visualization using the `Diwali Sales Data.csv` dataset.

## Project Structure

The project directory is organized as follows:

```plaintext
Diwali-Sales-Data-Python_Analysis/
│
├── data/
│   ├── raw/
│   │   └── Diwali Sales Data.csv  # Original uncleaned dataset
│
├── notebooks/
│   └── Diwali_sales.ipynb          # Jupyter notebook containing data analysis and visualization scripts
│
└── README.md                       # Project description and documentation

Project Documentation

1. Import Required Libraries

This project imports the necessary libraries for data processing and visualization:

import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting
import plotly.express as px  # For interactive visualizations
import time  # For time-related tasks
import re  # For string manipulation and pattern matching
import os  # For interacting with the operating system

2. Load the Dataset

Load the uncleaned dataset into a pandas DataFrame:

# Get the current working directory
pwd = os.getcwd()  # Storing the current directory path

# Load the uncleaned dataset into a pandas DataFrame
dataset = pd.read_csv(pwd + "/Diwali Sales Data.csv", encoding='ISO-8859-1')  # Reading the CSV file into a DataFrame

# Create a copy of the original dataset to work on
df = dataset.copy()  # Making a copy to avoid altering the original data

3. Initial Data Inspection

Inspect the dataset to understand its structure and content:

# Display 5 random samples from the dataset
df.sample(5)

# Display the first 5 rows of the dataset
df.head()

# Display the last 5 rows of the dataset
df.tail()

# Show the shape of the dataset
df.shape

# Show basic information about the dataset
df.info()

# Show the number of missing values for each column
df.isnull().sum()

# Drop duplicate rows if any
df = df.drop_duplicates()

# Drop unnecessary columns
df = df.drop(columns=["Status", "unnamed1"], axis=1)

# Remove rows with missing values
df = df.dropna()

# Convert the 'Amount' column to integer type
df["Amount"] = df["Amount"].astype(int)

4. Summary Statistics

Inspect summary statistics for numeric and categorical columns.

# Display value counts for object type columns
for i in df.select_dtypes(include="object").columns:
    print(f"Value Counts for {i}:\n", df[i].value_counts())

5. Exploratory Data Analysis (EDA)

The following EDA techniques are applied to gain insights into the dataset:

5.1 Gender Distribution

ax = sns.countplot(x="Gender", hue="Gender", data=df)
for bars in ax.containers:
    ax.bar_label(bars)
plt.show()

5.2 Total Amount by Gender

gender_amount_data = df.groupby(["Gender"])["Amount"].sum().reset_index()
sns.barplot(x="Gender", y="Amount", hue="Gender", data=gender_amount_data)
plt.show()

5.3 Age Group and Gender Distribution

gender_amount_data = df.groupby(["Gender"])["Amount"].sum().reset_index()
sns.barplot(x="Gender", y="Amount", hue="Gender", data=gender_amount_data)
plt.show()

etc.

6. Required Libraries
Ensure you have the following libraries installed in your Python environment:

pandas
numpy
seaborn
matplotlib
plotly
scikit-learn

You can install these libraries using pip:

pip install pandas numpy seaborn matplotlib plotly scikit-learn

7. Usage
To run the analysis, open the Jupyter notebook Diwali_sales.ipynb and execute each cell sequentially. Visualizations and insights will be generated based on the data analysis performed.

8. Author
Elkoudy
