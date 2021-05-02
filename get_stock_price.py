import pandas_datareader as web
import datetime
import numpy as np
import os
import random
import tensorflow as tf

# สำหรับ ไม่มี GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2021, 4, 30)
symbol="ADVANC"
df = web.DataReader(symbol+".BK", 'yahoo', start, end).reset_index()
df=df.drop_duplicates("Date").reset_index(drop=True)
df.to_csv("./model/csv/data_SET.csv",index=False)
df.head(5)