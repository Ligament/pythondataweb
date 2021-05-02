import datetime
import pandas_datareader as web
import numpy as np
from keras.models import load_model

backward=200

model = load_model("./model/ADVANC.h5")

start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2021, 4, 30) #วันที่วันนี้
symbol="ADVANC"
df = web.DataReader(symbol+".BK", 'yahoo', start, end).reset_index()
df=df.drop_duplicates("Date").reset_index(drop=True)
df=df[["High","Low","Open","Close"]]
df=df.dropna().reset_index(drop=True)
min1=df.min()
max1=df.max()
df=(df-min1)/(max1-min1)
df=df.dropna().reset_index(drop=True)
X=[]

for i in range(backward,df.shape[0]):
    Xt=np.array(df.iloc[i-backward:i])
    X.append(Xt)
X=np.array(X)
print((model.predict(X)*(max1["Close"]-min1["Close"])+min1["Close"])[-1])