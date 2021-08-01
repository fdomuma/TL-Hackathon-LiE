import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import os

# Loading data
print("Loading data...")
data= pd.read_csv("train.csv")
#meal_df = pd.read_csv("meal_info.csv")
#center_df = pd.read_csv("fulfilment_center_info.csv")
#data= df.merge(center_df,left_on = 'center_id', right_on = 'center_id',how="left")
#data= data.merge(meal_df,left_on = 'meal_id', right_on = 'meal_id',how="left")
print(data.head())

# Hot encoding
print("Hot encoding...")
def one_hot_encode(features_to_encode, dataset):
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(dataset[features_to_encode])

    encoded_cols = pd.DataFrame(encoder.transform(dataset[features_to_encode]),columns=encoder.get_feature_names())
    dataset = dataset.drop(columns=features_to_encode)
    for cols in encoded_cols.columns:
        dataset[cols] = encoded_cols[cols]
    return dataset

data = data.drop(["id"],axis=1)
#data = data.drop(["center_type", "category", "cuisine"],axis=1)
features_to_encode = ['meal_id',"center_id"]
data = one_hot_encode(features_to_encode, data)
y = data["num_orders"]
X= data.drop(["num_orders"],axis = 1)
print(X.head())

# Train, test partition
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Pipeline and fitting
print("Scaling data and creating model...")
RF_pipe = make_pipeline(StandardScaler(), RandomForestRegressor())
RF_pipe.fit(X_train, y_train)

# Predictions
print("Predicting result...")
RF_train_y_pred = RF_pipe.predict(X_test)
print(RF_pipe.score(X_test, y_test))
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_test, RF_train_y_pred)))

# Exporting
print("Exporting model...")
import pickle
#os.chdir("/kaggle/working/")
pickle.dump(RF_pipe, open("model.h5", 'wb'))
#os.chdir("/kaggle/input/")
