def removal_NA_weight1(row):
    
    if pd.isnull(row['Item_Weight']):
        return avg_weight1[row['Item_Identifier']]
    else:
        return row['Item_Weight']
    
def removal_NA_weight2(row):
    
    if pd.isnull(row['Item_Weight']):
        return avg_weight2[row['Item_Fat_Content'],row['Item_Type']]
    else:
        return row['Item_Weight']

    

def map_data(df):
    
    fat_content={"low fat":"Low Fat","LF":"Low Fat","reg":"Regular","Low Fat":"Low Fat","Regular":"Regular"}
    df['Item_Fat_Content']=df['Item_Fat_Content'].map(fat_content)
    return df

def process_data(df):
    new_df=df.copy()
    le=preprocessing.LabelEncoder()
    new_df['Item_Fat_Content']=le.fit_transform(new_df['Item_Fat_Content'])
    new_df['Item_Type']=le.fit_transform(new_df['Item_Type'])
    new_df['Outlet_Location_Type']=le.fit_transform(new_df['Outlet_Location_Type'])
    new_df['Outlet_Size']=le.fit_transform(new_df['Outlet_Size'])
    new_df['Outlet_Type']=le.fit_transform(new_df['Outlet_Type'])
    new_df['Item_Identifier']=le.fit_transform(new_df['Item_Identifier'])
    new_df['Outlet_Identifier']=le.fit_transform(new_df['Outlet_Identifier'])
    return new_df




def ML(df1,df2):
    X=df1.drop(['Item_Outlet_Sales'],axis=1).values
    Y=df1['Item_Outlet_Sales'].values
    #feature_train,feature_test,label_train,label_test=train_test_split(X,Y,test_size=0.2)
    reg=LinearRegression()
    reg.fit(X,Y)
    X1=df2.values
    pred=reg.predict(X1)
    print(reg.score(X1,pred))
    #i=5
    #try:
     #   for feature, target in zip(X, Y):
      #      plt.scatter( feature[i], target, color="b" ) 
    #except ValueError:
     #   pass
    return pred
        

    
    
    
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

sdf1=pd.read_csv("Train(Sales).csv")
sdf2=pd.read_csv("Test(Sales).csv")

avg_weight1=sdf1.groupby(['Item_Identifier'])['Item_Weight'].mean()
avg_weight2=sdf1.groupby(['Item_Fat_Content','Item_Type'])['Item_Weight'].mean()

sdf1=map_data(sdf1)
sdf2=map_data(sdf2)

sdf1['Item_Weight'] =sdf1.apply(removal_NA_weight1, axis=1)
sdf2['Item_Weight'] =sdf2.apply(removal_NA_weight1, axis=1)

sdf1['Item_Weight'] =sdf1.apply(removal_NA_weight2, axis=1)
sdf2['Item_Weight'] =sdf2.apply(removal_NA_weight2, axis=1)

sdf1['Outlet_Size'].fillna('Small',inplace=True)
sdf2['Outlet_Size'].fillna('Small',inplace=True)

ndf1=process_data(sdf1)
ndf2=process_data(sdf2)

k=ML(ndf1,ndf2)
