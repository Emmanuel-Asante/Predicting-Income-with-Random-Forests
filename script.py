def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load data
income_data = pd.read_csv("income.csv", header = 0, delimiter = ", ")

# Print out the first row of income_data
print(income_data.iloc[0])

# create label
labels = income_data[["income"]]

# Create sex-int column
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)

# create country-int column
income_data["country-int"] = income_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)

# Select columns
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]]

# Split data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

# create a random forest model
forest = RandomForestClassifier(random_state = 1)

# Train the model
forest.fit(train_data, train_labels)

# Calculate the model's accuracy
print(forest.score(test_data, test_labels))

# Print out the values in the native-country column
#print(income_data["native-country"].value_counts())

# Create a decision tree model object
tree = DecisionTreeClassifier()

# Train the tree model
tree.fit(train_data, train_labels)

# Calculate the tree model's accuracy
print(tree.score(test_data, test_labels))

# Find the relevant column
print(forest.feature_importances_)