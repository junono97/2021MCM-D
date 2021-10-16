#https://blog.csdn.net/u012967763/article/details/79175946

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest,chi2,SelectFromModel,f_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from xgboost.sklearn import XGBClassifier,XGBRegressor

def SelectFromModel_xgb(x_train, y_train,x_test, y_test):
    xgb_s=SelectFromModel(XGBRegressor(),max_features=20)
    x_train_new =xgb_s.fit_transform(x_train, y_train)
    x_test_new =xgb_s.transform(x_test)
    build_model_xgb(x_train_new, y_train,x_test_new, y_test)
    return x_train.columns[xgb_s.get_support()]
def build_model_xgb(x_train, y_train,x_test, y_test):
    print("--------build_model_df-------")
    xgb = XGBRegressor()
    xgb.fit(x_train, y_train)
    print("xgb,Score:",xgb.score(x_test,y_test))
    
"""导入数据集"""
x=pd.read_csv("x.csv")
y=pd.read_csv("y1.csv")
x_data=x.iloc[:,1:]
y_data=y.iloc[:,2]

# 循环验证20次
# for i in range(20):
#     x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=0)
#     strr=SelectFromModel_xgb(x_train, y_train,x_test, y_test)
#     print(strr)
#     np.savetxt("epoch/%d.txt"%i,strr,fmt="%s")

dict={}
# 寻找最优20个解
for i in range(20):
    contents=np.loadtxt("epoch/%d.txt"%i,dtype=str)
    for content in contents:
        if content not in dict:
            dict[content]=1
        else:
            dict[content]+=1
print(sorted(dict.items(), key = lambda kv:(kv[1], kv[0]),reverse=True))
ans=[]
for k,v in sorted(dict.items(), key = lambda kv:(kv[1], kv[0]),reverse=True):
    ans.append(k)
#np.savetxt("20fea.txt",np.array(ans)[:20],fmt="%s")