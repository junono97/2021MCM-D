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


def build_model_df(x_train, y_train,x_test, y_test):
    print("--------build_model_df-------")
    dt = DecisionTreeRegressor()
    dt.fit(x_train, y_train)
    print("DF,Score:",dt.score(x_test,y_test))
    
def build_model_ab(x_train, y_train,x_test, y_test):
    print("--------build_model_ab-------")
    ab=AdaBoostRegressor()
    ab.fit(x_train, y_train)
    print("Adaboost,Score:",ab.score(x_test,y_test))
    
def build_model_ridge(x_train, y_train,x_test, y_test):
    print("--------build_model_df-------")
    ridge = Ridge()#(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001)
    ridge.fit(x_train, y_train)
    print("ridge,Score:",ridge.score(x_test,y_test))

def build_model_xgb(x_train, y_train,x_test, y_test):
    print("--------build_model_df-------")
    xgb = XGBRegressor()#(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001)
    xgb.fit(x_train, y_train)
    print("xgb,Score:",xgb.score(x_test,y_test))
    
def SelectFromModel_ridge(x_train, y_train,x_test, y_test):
    ridge_s=SelectFromModel(Ridge(),max_features=20)
    x_train_new =ridge_s.fit_transform(x_train, y_train)
    x_test_new =ridge_s.transform(x_test)
    #print("ridge select",x_train.columns[ridge_s.get_support()])
    build_model_ridge(x_train_new, y_train,x_test_new, y_test)

def SelectFromModel_xgb(x_train, y_train,x_test, y_test):
    xgb_s=SelectFromModel(XGBRegressor(),max_features=20)
    x_train_new =xgb_s.fit_transform(x_train, y_train)
    x_test_new =xgb_s.transform(x_test)
    #print("ridge select",x_train.columns[xgb_s.get_support()])
    build_model_xgb(x_train_new, y_train,x_test_new, y_test)
    
def SelectFromModel_df(x_train, y_train,x_test, y_test):
    df_s=SelectFromModel(DecisionTreeRegressor(),max_features=20)
    x_train_new =df_s.fit_transform(x_train, y_train)
    x_test_new =df_s.transform(x_test)
    #print("df select:",x_train.columns[df_s.get_support()])
    build_model_df(x_train_new, y_train,x_test_new, y_test)
    
def SelectFromModel_ab(x_train, y_train,x_test, y_test):
    ab_s=SelectFromModel(AdaBoostRegressor(),max_features=20)
    x_train_new =ab_s.fit_transform(x_train, y_train)
    x_test_new =ab_s.transform(x_test)
    #print("ab select",x_train.columns[ab_s.get_support()])
    build_model_ab(x_train_new, y_train,x_test_new, y_test)
    
"""导入数据集"""
x_data=np.load("pearsonr.npy")
print(x_data.shape)
y=pd.read_csv("y1.csv")
y_data=y.iloc[:,2]
print(y_data.shape)
"""划分训练/测试集"""
# 4:1划分训练集测试集,种子为1.
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=0)
"""建模"""
print(".................ALL FEATURE.................")
build_model_df(x_train, y_train,x_test, y_test)
build_model_ab(x_train, y_train,x_test, y_test)
build_model_ridge(x_train, y_train,x_test, y_test)
build_model_xgb(x_train, y_train,x_test, y_test)