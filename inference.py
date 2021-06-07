import pickle
import numpy as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy
import joblib


model = pickle.load(open('model.pkl','rb'))
df = pd.read_csv("E:\\salespred\\sales.csv")

def predict(df):
    cols_when_model_builds = model.get_booster().feature_names
    salesdata = df[cols_when_model_builds]

    salesdata=df.iloc[:,0:-1]
    print(salesdata)
    salesdata.isnull().sum()

    outlet_size_mode = salesdata.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

    missing_values = salesdata['Outlet_Size'].isnull()

    salesdata.loc[missing_values, 'Outlet_Size'] = salesdata.loc[missing_values,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
    salesdata['Item_Fat_Content'].value_counts()
    salesdata.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
    salesdata['Item_Fat_Content'].value_counts()
    #label encoding
    d = joblib.load('enc.sav')

    for i in d:
      encoder = LabelEncoder()
      encoder.classes_ = d[i]
      salesdata[i] = encoder.transform(salesdata[i])

    salesdata=salesdata.iloc[0:,:];
    predictions = model.predict(salesdata)
    return list(predictions)


print(predict(df))