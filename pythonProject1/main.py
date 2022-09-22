import numpy as np
import pandas as pd
import knn as cl
import cross_validation as val
path1='red.csv'#已分类样本
path2='test.csv'#待分类样本
data=pd.read_csv(path1)
test=pd.read_csv(path2)
row=test.shape[0]#待测样本个数
line=test.shape[1]#待测样本特征数量
k=val.validaton(data)
print(k)
for i in range(row):
    print(i)
    feature=test.iloc[i,:]
    lable=cl.classify(feature,k,data)
    test.iloc[i,line-1]=lable
test.to_csv('result.csv')









