import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,LabelBinarizer, StandardScaler
from sklearn.pipeline import FeatureUnion
# Models import
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# Cross-validation feature
from sklearn.model_selection import cross_val_score
# Evaluation import
from sklearn.metrics import mean_squared_error
# Grid Search
from sklearn.model_selection import GridSearchCV


PATH = "data/handson-ml-master/datasets/housing/housing.csv"


def get_data():
    df = pd.read_csv(PATH)
    # print(df.info())
    return df


if os.getcwd().endswith('python'):
    df = get_data()
else:
    os.chdir("..")
    df = get_data()

# Create income categories to limit the number of categories
#  and rounding up
df["income_cat"] = np.ceil(df["median_income"] / 1.5)
df["income_cat"].where(df["income_cat"] < 5, 5.0, inplace=True)

# Stratified sampling based on the income category
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                               random_state=42)

for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

# Remove the income_cat attribute to revert back to the original state
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# Strategy for imputing missing value: use median
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing_num = housing.drop("ocean_proximity", axis=1)

encoder = LabelEncoder()

# Encoding
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)


# Custom Transformers
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household,
                         population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# Class for selecting desired dataframe
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

# class for improving BinaryLabelizer to work in Pipeline
class ImprovedLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

# Feature Scaling and transformation pipelines
# Complete pipeline including cat attributes
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', ImprovedLabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

# Pipe the dataset for processing
housing_prep = full_pipeline.fit_transform(housing)

"""
Training and Evaluating on the training set
"""
# Linear model

lin_reg = LinearRegression()
lin_reg.fit(housing_prep, housing_labels)

housing_predictions = lin_reg.predict(housing_prep)

lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("RMSE for linear model: ", lin_rmse)

# Decision Tree model
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prep, housing_labels)

housing_predictions = tree_reg.predict(housing_prep)

tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("RMSE for Decision Tree model: ", tree_rmse)

# Random Forest model
rf_reg = RandomForestRegressor()
rf_reg.fit(housing_prep, housing_labels)

housing_predictions = rf_reg.predict(housing_prep)

rf_mse = mean_squared_error(housing_labels, housing_predictions)
rf_rmse = np.sqrt(rf_mse)
print("RMSE for Random Forest model: ", rf_rmse)


# Using cross-validation
def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

lin_scores = cross_val_score(lin_reg, housing_prep, housing_labels,
            scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

tree_scores = cross_val_score(tree_reg, housing_prep, housing_labels,
            scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

rf_scores = cross_val_score(rf_reg, housing_prep, housing_labels,
            scoring="neg_mean_squared_error", cv=10)
rf_rmse_scores = np.sqrt(-rf_scores)

display_scores(lin_rmse_scores)
display_scores(tree_rmse_scores)
display_scores(rf_rmse_scores)

# Fine tuning the model using Grid search
param_grid = [
    {'n_estimators': [3, 10, 30],
    'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False],
    'n_estimators': [3, 10],
    'max_features': [2, 4, 6]}
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg,
                           param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(housing_prep, housing_labels)

print(grid_search.best_params_)

# Listing the evaluation scores together with the features
cv_res = grid_search.cv_results_

for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(np.sqrt(-mean_score), params)

# Looking at the best model and their attributes importance
features_weight = grid_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(features_weight, attributes), reverse=True)

# Evaluate the choosen model on the test set

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(final_rmse)