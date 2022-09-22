import knn as cl
import numpy as np
import pandas as pd
import matplotlib as mp


def validaton(train):
    precision=np.zeros(21)#记录每个K值的验证准确率
    col=train.shape[0]#记录列数
    line=train.shape[1]#记录行数
    size=int(col/5)
    idx=np.arange(col)
    for k in range(4,5):
        correct=0
        for i in range(1,5):
            start,stop=size*(i-1),size*i

            temp=train
            for j in range(int(start),int(stop)):
                temp.drop([j])
            temp=temp.reset_index(drop=True)
            for j in range(start,stop):
                lable=cl.classify(train.iloc[j,:],k,temp)
                if(lable==train.iloc[j,line-1]):
                      correct+=1
                      print(correct)
        precision[k]=correct/(size*5)
        print(precision[k])
    precision2=pd.Series(precision)
    precision2=precision2.sort_values(ascending=False)
    return   (precision2.index)[0]

