import pandas as pd
import numpy as np

# Load the Melbourne housing data
melbourne_data = pd.read_csv('melb_data.csv')

# Select relevant columns and drop rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)

# Define the target variable and features
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X=melbourne_data[melbourne_features]

# Explore the data
X.head()
X.describe()

# Train a Decision Tree Regressor training the model
from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


# Evaluate the model using Mean Absolute Error model validation methoud 

from sklearn.metrics import mean_absolute_error

predicted_home_price=melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_price)

# Split the data into two set for  training and validation sets for better evaluation
#random_state to ensure same split every time we run the code

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

#overfitting issue and underfitting issue solved by tuning the model using max_leaf_nodes parameter

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Compare MAE with different values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# Try using a Random Forest Regressor to improve accuracy
#random forest generally performs better than decision trees
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
forest_val_predictions = forest_model.predict(val_X)
print(mean_absolute_error(val_y, forest_val_predictions))
