import pandas as pd
import numpy as np
def classify(t,k,data):
    row=data.shape[0]#样本数量，行
    line=data.shape[1]#样本特征与标签，列
    dis = np.zeros(row);
    dis1 = pd.Series(dis);  # 保存测试样本与数据集中每个样本之间的距离
    for i in range(row):
        sum=0
        for j in range(1,line-1):
            sum+=(t[j]-data.iloc[i,j])**2
        dis1[i]=sum**(1/2)#计算欧式距离
    dis2=dis1.sort_values()#根据距离升序排列
    dis3=dis2.index
    #统计距离最近的k个样本的标签
    lable=pd.Series(np.zeros(k))
    for j in range(k):
        lable[j]=data.iloc[dis3[j],line-1]
    count=lable.value_counts()
    return (count.index)[0]
    #return (count.index)[0]