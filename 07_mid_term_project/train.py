# Import libraries

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# Output file

output_file = 'model_xgb.bin'

print("Reading the file....")
# reading the file

df = pd.read_csv("insurance_claims.csv")

print("Preprocessing....")
df = df.replace('?', np.NaN)


# Among all collision type **Rear Collision is the most common** as collision type can be None
# Hence we will replace the NaN value with **Rear Collision**


df['collision_type'] = df['collision_type'].fillna('Rear Collision')
df['collision_type'].value_counts()

# There might be case were there is **no property damange** and **no police report available**
# hence, we will replace that with **NO**


df['property_damage'] = df['property_damage'].fillna('NO')
df['police_report_available'] = df['police_report_available'].fillna('NO')

# **Changing the target label to binary, Fraud_reported - Yes: 1 and Fraud_reported - No: 0**


df['fraud_reported'] = df['fraud_reported'].replace(('Y', 'N'), (1, 0))


# Deriving the age of the vehicle based on the year value 

df['vehicle_age'] = 2018 - df['auto_year'] 


# removing unnecessary features

df = df.drop(columns = [
    'policy_number', 
    'insured_zip',
    'policy_bind_date', 
    'incident_date', 
    'incident_location',
    'auto_year'])


# features

categorical = ['policy_state', 'policy_csl', 'insured_sex', 'insured_education_level',
       'insured_occupation', 'insured_hobbies', 'insured_relationship',
       'incident_type', 'collision_type', 'incident_severity',
       'authorities_contacted', 'incident_state', 'incident_city',
       'property_damage', 'police_report_available', 'auto_make',
       'auto_model']

numerical = ['months_as_customer', 'age', 'policy_deductable',
       'policy_annual_premium', 'umbrella_limit', 'capital-gains',
       'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
       'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
       'fraud_reported', 'vehicle_age']

print("Splitting the data....")
# Splitting the Data


df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=11)

df_train_full = df_train_full.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train_full = df_train_full.fraud_reported.values
y_train = df_train.fraud_reported.values
y_val = df_val.fraud_reported.values
y_test = df_test.fraud_reported.values

del df_train_full['fraud_reported']
del df_train['fraud_reported']
del df_val['fraud_reported']
del df_test['fraud_reported']


#  DictVectorizer

train_full_dicts = df_train_full.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train_full = dv.fit_transform(train_full_dicts)
X_test = dv.transform(test_dicts)

features = dv.get_feature_names()

dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

print("Training the final model....")
# XGBoost Final Model:

xgb_params = {
    'eta': 0.01, 
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 5,
    
    'objective': 'binary:logistic',
    'nthread': 4,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain_full, num_boost_round=75)

print("Saving the model....")
# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


