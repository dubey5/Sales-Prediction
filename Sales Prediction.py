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

train=pd.read_csv("Train(Sales).csv")
test=pd.read_csv("Test(Sales).csv")

train['source']='train'
test['source']='test'

combine=pd.concat([train,test],ignore_index=True)

avg_weight=combine.groupby(['Item_Identifier'])['Item_Weight'].mean()
avg_visibility=combine.groupby(['Item_Identifier'])['Item_Visibility'].mean()

combine=map_data(combine)


#ndf1=process_data(sdf1)
#ndf2=process_data(sdf2)

#k=ML(ndf1,ndf2)
