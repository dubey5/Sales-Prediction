def removal_NA_weight(row):
    
    if pd.isnull(row['Item_Weight']):
        return avg_weight[row['Item_Identifier']]
    else:
        return row['Item_Weight']
    
def removal_zero_visib(row):
    
    if row['Item_Visibility']==0:
        return avg_visibility[row['Item_Identifier']]
    else:
        return row['Item_Visibility']


def map_data(df):
    df['Item_Weight'] =df.apply(removal_NA_weight, axis=1)
    df['Outlet_Size'].fillna('Small',inplace=True)
    df['Item_Visibility'] =df.apply(removal_zero_visib, axis=1)
    df['Item_Fat_Content']=df['Item_Fat_Content'].replace({"low fat":"Low Fat","LF":"Low Fat","reg":"Regular"})
    df['Outlet_Years']=2013-df['Outlet_Establishment_Year']
    return df

def process_data(df):
    new_df=df.copy()
    le=preprocessing.LabelEncoder()
    cat_var=['Item_Fat_Content','Item_Type','Outlet_Location_Type','Outlet_Size','Outlet_Type']
    for i in cat_var:
        new_df[i]=le.fit_transform(new_df[i])
    new_df.drop(['Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
    return new_df




def ML(df):
    
    train_set=df[df['source']=="train"]
    test_set=df[df['source']=="test"]
    
    train_f=train_set.drop(['source','Item_Outlet_Sales'],axis=1).values
    train_l=train_set['Item_Outlet_Sales'].values
    
    test_f=test_set.drop(['Item_Outlet_Sales','source'],axis=1).values
    
    
    reg=LinearRegression()
    reg.fit(train_f,train_l)
    
    pred=reg.predict(test_f)
    print(reg.score(test_f,pred))
    
        

    
    
    
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

train=pd.read_csv("Train(Sales).csv")
test=pd.read_csv("Test(Sales).csv")

train['source']='train'
test['source']='test'

combine=pd.concat([train,test],ignore_index=True)

avg_weight=combine.groupby(['Item_Identifier'])['Item_Weight'].mean()
avg_visibility=combine.groupby(['Item_Identifier'])['Item_Visibility'].mean()

combine=map_data(combine)

ndf=process_data(combine)

ML(ndf)


