import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
import sys

pd.set_option('display.max_rows', 250)  # if print: show all rows
pd.set_option('display.max_columns', None)  # if print show all columns

print("### info: read csv file")
weather = pd.read_csv("weatherAUS.csv")  # read csv data into pandas data frame
# print(matrix_describe)

# restructure Date field into Year, Month, Day and drop Date
print("### info: extract and drop date")
weather['Date'] = pd.to_datetime(weather.Date, format="%Y-%m-%d")
weather['Year'] = weather['Date'].dt.year  # get year
weather['Month'] = weather['Date'].dt.month  # get month
weather['Day'] = weather['Date'].dt.day  # get day
weather.drop(labels=['Date'], axis=1, inplace=True)

#   we only work with years 2013 and 2018
print("### info: filter for years 2013 and 2018")
weather = weather[weather.Year.isin([2013, 2018])]
# # plot example
# plt.hist(weather.loc[weather.Location == 'Adelaide', 'MinTemp'], 20)
# plt.show()

# ############################################
# delete columns that have Na / NaN > 70%
# ############################################
print("### info: check for columns to delete")
# get prop of  Na values per column
prop_na_per_col = weather.isna().sum() / len(weather)
cols_to_delete = []
for i in prop_na_per_col.index:
    if prop_na_per_col[i] > 0.7:
        cols_to_delete.append(i)
# drop variables evaporation and sunshine
# axis = 1 : delete columns, inplace=True: replace current dataframe
print("### info: drop ", cols_to_delete, "from data frame")
weather.drop(labels=cols_to_delete, axis=1, inplace=True)

# ############################################
# delete rows that have Na / NaN > 70%
# delete location data if it generally produces bad data
# ############################################
print("### info: identify rows to be deleted")
# get prop of Na values per row
i_rows_to_delete = []
for i in range(len(weather.index)):
    prop = weather.iloc[i].isnull().sum() / weather.shape[1]
    if prop > 0.70:
        i_rows_to_delete.append(i)

print("Total rows to delete: ", len(i_rows_to_delete))
print("Prop rows to delete: ", len(i_rows_to_delete) / weather.shape[0])

# get prop of rows to delete per location -> check if specific location produces bad data
print("### info: identify locations that produce bad data")
locations_to_be_deleted = []
for location in weather['Location'].unique():
    location_rows = weather.iloc[i_rows_to_delete]  # get all rows to be deleted
    location_rows = location_rows.loc[
        location_rows["Location"] == location]  # of those rows get the ones with current location
    prop = len(location_rows) / len(weather.loc[weather["Location"] == location])  # get prop of bad rows per location
    if prop > 0.5:
        locations_to_be_deleted.append(location)
        print("prop rows to delete ", location, " : ", prop)
# -> no location needs to be dropped
print("### info: number of locations to be deleted: ", len(locations_to_be_deleted))
# drop the identifies rows
print("### info: drop identified ", len(i_rows_to_delete), "rows")
rows_before = weather.shape[0]
weather.drop(labels=weather.index[i_rows_to_delete], axis=0, inplace=True)
print("num rows after: ", weather.shape[0], "should equal: ", rows_before - len(i_rows_to_delete))

# ###########################################
# handle NaN values
# ###########################################
# float values
grouped_weather = weather.groupby('Location').describe(include=[np.number])
matrix_describe = grouped_weather.loc[:, grouped_weather.columns.get_level_values(1).isin({"mean", "50%"})]

for col in ["MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
            "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
            "Temp9am", "Temp3pm"]:

    if weather[col].isna().any():
        for location in weather['Location'].unique():
            # always calc median grouped by location
            location_col_median = matrix_describe.loc[location, ((matrix_describe.columns.get_level_values(1) == "50%")
                                                                 & (matrix_describe.columns.get_level_values(
                        0) == col))]
            if location_col_median.isna().bool():
                weather.loc[weather['Location'] == location, col] = 0
            else:
                if str.startswith(col, "Cloud"):
                    location_col_median[0] = round(location_col_median[0])
                weather.loc[weather['Location'] == location, col] = location_col_median[0]
    else:
        print(col, ": no NaN values found")

# sys.exit("End here")
# ############################################
# perform one hot encoding
# NaNs are ignored
# one hot encoding is the only possibility to handle categorical data in sklearn decision trees
# variables to encode: location, windDir (all)
# ############################################
for col in ['Location', "WindGustDir", "WindDir9am", "WindDir3pm"]:
    encoded_columns = pd.get_dummies(weather[col], prefix=col, drop_first=True)
    weather = weather.join(encoded_columns).drop(col, axis=1)

# ############################################
# encode binary data
# ############################################
for col in ['RainToday', 'RainTomorrow']:
    weather.loc[weather[col] == "No", col] = 0
    weather.loc[weather[col].isna(), col] = 0
    weather.loc[weather[col] == "Yes", col] = 1
    weather[col] = weather[col].astype(int)

print(weather.isna().any().any(), " should equal False")
# #############################################
# A2 - decision trees
# #############################################
n = weather.shape[0]  # number of rows

X_train, X_test, y_train, y_test = train_test_split(weather.drop(["RainTomorrow"], axis=1), weather.RainTomorrow,
                                                    test_size=0.2)

clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(X=X_train, y=y_train)

plt.figure(figsize=(25, 10))
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(50, 50), dpi=300)
tree.plot_tree(clf,
               feature_names=weather.columns,
               filled=True,
               rounded=True,
               fontsize=10)
fig.savefig('tree.png')
