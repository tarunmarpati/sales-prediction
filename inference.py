import pickle
import numpy as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder


model = pickle.load(open('model.pkl','rb'))


def predict(df):
    cols_when_model_builds = model.get_booster().feature_names
    salesdata = df[cols_when_model_builds]

    salesdata=df.iloc[:,0:-1]
    # print(salesdata)
    salesdata.isnull().sum()

    outlet_size_mode = salesdata.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

    missing_values = salesdata['Outlet_Size'].isnull()

    salesdata.loc[missing_values, 'Outlet_Size'] = salesdata.loc[missing_values,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
    salesdata['Item_Fat_Content'].value_counts()
    salesdata.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
    salesdata['Item_Fat_Content'].value_counts()
    #label encoding
    encoder = LabelEncoder()
    salesdata['Item_Identifier'] = encoder.fit_transform(salesdata['Item_Identifier'])

    salesdata['Item_Fat_Content'] = encoder.fit_transform(salesdata['Item_Fat_Content'])

    salesdata['Item_Type'] = encoder.fit_transform(salesdata['Item_Type'])

    salesdata['Outlet_Identifier'] = encoder.fit_transform(salesdata['Outlet_Identifier'])

    salesdata['Outlet_Size'] = encoder.fit_transform(salesdata['Outlet_Size'])

    salesdata['Outlet_Location_Type'] = encoder.fit_transform(salesdata['Outlet_Location_Type'])

    salesdata['Outlet_Type'] = encoder.fit_transform(salesdata['Outlet_Type'])
    salesdata=salesdata.iloc[1:,:];
    # print(salesdata.head())
    # print(np.shape(salesdata))
    predictions = model.predict(salesdata)
    return list(predictions)

