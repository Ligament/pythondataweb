import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.callbacks import EarlyStopping

df=pd.read_csv("./model/csv/data_SET.csv", usecols=["Date","High","Low","Open","Close"])

forward=5
df["output"]=df["Close"].shift(-forward)
df=df.drop(columns="Date").dropna().reset_index(drop=True)
# df=df[["Open","Close","output"]]
min1=df.min()
max1=df.max()
df=(df-min1)/(max1-min1)
df=df.dropna().reset_index(drop=True)
backward=200
df1=df.drop(columns="output")
X=[]
y=[]

for i in range(backward,df.shape[0]):
    Xt=np.array(df1.iloc[i-backward:i])
    X.append(Xt)
    yt=df.loc[i-1,"output"]
    y.append([yt])
X=np.array(X)
y=np.array(y)
del df,df1,Xt,yt
n_train=int(X.shape[0]*0.8)
print("dimension: ",X.shape,y.shape)
X_train=X[:n_train]
X_test=X[n_train:]
y_train=y[:n_train]
y_test=y[n_train:]
y_train.shape

model = Sequential()
model.add(LSTM(60, input_shape=(backward, X.shape[2]),activation="tanh",use_bias=True,return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(40, input_shape=(backward, 60),activation="relu",use_bias=True,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(X_train,y_train, epochs=1000, batch_size=X_train.shape[0], verbose=0, validation_data=(X_test, y_test),callbacks=[es])
y_train=y_train*(max1["output"]-min1["output"])+min1["output"]
y_test=y_test*(max1["output"]-min1["output"])+min1["output"]
y_train1=model.predict(X_train)*(max1["output"]-min1["output"])+min1["output"]
y_test1=model.predict(X_test)*(max1["output"]-min1["output"])+min1["output"]

model.save("./model/ADVANC.h5")

print(np.mean(np.abs(y_train-y_train1)/y_train),np.mean(np.abs(y_train-y_train1)))
print(np.mean(np.abs(y_test-y_test1)/y_test),np.mean(np.abs(y_test-y_test1)))

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(range(y_test.shape[0]),y_test.transpose()[0]-y_test1.transpose()[0],"o")
axs[0, 1].plot(range(y_test.shape[0]),y_test.transpose()[0],"b")
axs[0, 1].plot(range(y_test.shape[0]),y_test1.transpose()[0],"r")
axs[1, 0].plot(range(y_train.shape[0]),y_train.transpose()[0]-y_train1.transpose()[0],"o")
axs[1, 1].plot(range(y_train.shape[0]),y_train.transpose()[0],"b")
axs[1, 1].plot(range(y_train.shape[0]),y_train1.transpose()[0],"r")
fig.tight_layout()
plt.show()