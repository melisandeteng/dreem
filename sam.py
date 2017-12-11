import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense, Dropout,Lambda
# from bisect import bisect
# import keras.backend as K
# import pywt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder

n_f1 = 600
n_f2 = 100

data = pd.read_csv('C:/Users/SAMUEL/Documents/MLC/train/data/datasets/supelec/train.csv')
list_col = data.columns

list_eeg = [x for x in data.columns if 'eeg' in x]
list_resp = {}
list_resp['x'] = [x for x in data.columns if 'respiration_x' in x]
list_resp['y'] = [x for x in data.columns if 'respiration_y' in x]
list_resp['z'] = [x for x in data.columns if 'respiration_z' in x]
list_user = np.unique(data['user'])
user = {}
for i in list_user:
    user[i] = data[data.user == i].index
l = ['x', 'y', 'z']

pca1 = PCA(n_components=n_f1)
pca2 = {}

enc = OneHotEncoder()
enc.fit(data['user'].as_matrix().reshape(50000, 1))

f1 = pca1.fit_transform(scale(data[list_eeg].as_matrix(), axis=0))
f2 = {}
for i in l:
    pca2[i] = PCA(n_components=n_f2)
    f2[i] = pca2[i].fit_transform(scale(data[list_resp[i]].as_matrix(), axis=0))

f_tr = np.concatenate([f1] + [f2[i] for i in l], axis=1)
f_tr = np.concatenate([f_tr, data[['time_previous', 'number_previous', 'time']].as_matrix(),
                       enc.transform(data['user'].as_matrix().reshape(50000, 1)).toarray()], axis=1)

t_tr = data['power_increase'].as_matrix()
del (data)

#
# f1=np.apply_along_axis(lambda x: np.append(np.fft.fft(x,n_f1).real,np.fft.fft(x,n_f1).imag),1,data[list_eeg].as_matrix())
# f2={}
# for i in l:
#    f2[i]=np.apply_along_axis(lambda x: np.append(np.fft.fft(x,n_f2).real,np.fft.fft(x,n_f2).imag),1,data[list_resp[i]].as_matrix())
#
# f_tr=np.concatenate([f1]+[f2[i] for i in l],axis=1)
# t_tr=data['power_increase'].as_matrix()
#
# del(data)
#
# data=pd.read_csv('C:/Users/SAMUEL/Documents/MLC/train/data/datasets/supelec/train.csv',skiprows=25000,header=None,names=list_col)
#
# f1=np.apply_along_axis(lambda x: np.append(np.fft.fft(x,n_f1).real,np.fft.fft(x,n_f1).imag),1,data[list_eeg].as_matrix())
# f2={}
# for i in l:
#    f2[i]=np.apply_along_axis(lambda x: np.append(np.fft.fft(x,n_f2).real,np.fft.fft(x,n_f2).imag),1,data[list_resp[i]].as_matrix())
#
# f_te=np.concatenate([f1]+[f2[i] for i in l],axis=1)
# t_te=data['power_increase'].as_matrix()
#
# del(data)



# f1=np.apply_along_axis(lambda x: np.append(pywt.cwt(x,range(1,n_f1+1),'gaus2')),1,data[list_eeg].as_matrix())
# f2={}
# for i in l:
#    f2[i]=np.apply_along_axis(lambda x: np.append(pywt.cwt(x,range(1,n_f2+1),'gaus2')),1,data[list_resp[i]].as_matrix())
#
# f_tr=np.concatenate([f1]+[f2[i] for i in l],axis=1)
# t_tr=data['power_increase'].as_matrix()
#
# del(data)
#
# data=pd.read_csv('C:/Users/SAMUEL/Documents/MLC/train/data/datasets/supelec/train.csv',skiprows=25000,header=None,names=list_col)
#
# f1=np.apply_along_axis(lambda x: np.append(pywt.cwt(x,range(1,n_f1+1),'gaus2')),1,data[list_eeg].as_matrix())
# f2={}
# for i in l:
#    f2[i]=np.apply_along_axis(lambda x: np.append(pywt.cwt(x,range(1,n_f2+1),'gaus2')),1,data[list_resp[i]].as_matrix())
#
# f_te=np.concatenate([f1]+[f2[i] for i in l],axis=1)
# t_te=data['power_increase'].as_matrix()
#
# del(data)

# pca1 = PCA(n_components=n_f1)
# pca2={}
#
# f1=pca1.fit_transform(scale(data[list_eeg].as_matrix(),axis=0))
# f2={}
# for i in l:
#    pca2[i]=PCA(n_components=n_f2)
#    f2[i]=pca2[i].fit_transform(scale(data[list_resp[i]].as_matrix(),axis=0))
#
# f_tr=np.concatenate([f1]+[f2[i] for i in l],axis=1)
# f_tr=np.concatenate([f_tr,data[['time_previous', 'number_previous', 'time']].as_matrix()],axis=1)
# t_tr=data['power_increase'].as_matrix()
#
# del(data)
#
# data=pd.read_csv('C:/Users/SAMUEL/Documents/MLC/train/data/datasets/supelec/train.csv',skiprows=25000,header=None,names=list_col)
#
# f1=pca1.transform(scale(data[list_eeg].as_matrix(),axis=0))
# f2={}
# for i in l:
#    f2[i]=pca2[i].transform(scale(data[list_resp[i]].as_matrix(),axis=0))
#
# f_te=np.concatenate([f1]+[f2[i] for i in l],axis=1)
# t_te=data['power_increase'].as_matrix()
# f_te=np.concatenate([f_te,data[['time_previous', 'number_previous', 'time']].as_matrix()],axis=1)
#
# del(data)


# f1=scale(data[list_eeg].as_matrix(),axis=0)
# f2={}
# for i in l:
#    f2[i]=scale(data[list_resp[i]].as_matrix(),axis=0)
#
# f_tr=np.concatenate([f1]+[f2[i] for i in l],axis=1)
##f_tr=np.concatenate([f_tr,data[['time_previous', 'number_previous', 'time']].as_matrix()],axis=1)
# t_tr=data['power_increase'].as_matrix()
#
# del(data)
#
# data=pd.read_csv('C:/Users/SAMUEL/Documents/MLC/train/data/datasets/supelec/train.csv',skiprows=25000,header=None,names=list_col)
#
# f1=scale(data[list_eeg].as_matrix(),axis=0)
# f2={}
# for i in l:
#    f2[i]=scale(data[list_resp[i]].as_matrix(),axis=0)
#
# f_te=np.concatenate([f1]+[f2[i] for i in l],axis=1)
# t_te=data['power_increase'].as_matrix()
##f_te=np.concatenate([f_te,data[['time_previous', 'number_previous', 'time']].as_matrix()],axis=1)
#
# del(data)

"""
for k in range(3):
    plt.figure()
    plt.plot(data[list_eeg].iloc[k].as_matrix())



for k in range(3):
    for i in range(3):
        plt.figure()
        plt.plot(data[list_resp[l[i]]].iloc[k].as_matrix())
"""

# f_tr=f_tr[:,:800]
# f_te=f_te[:,:800]

# model=Sequential()
# model.add(Dense(units=500, activation='tanh',input_dim=3200))
# model.add(Dropout(0.5))
# model.add(Dense(units=500, activation='tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(units=500, activation='tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(units=1, activation='linear'))
#
# model.compile(loss='mse', optimizer='rmsprop')
# model.fit(f_tr,t_tr,epochs=5,batch_size=50)
#
# print(model.predict(f_te),t_te,model.evaluate(f_te,t_te))

# def conv(x):
#    res=np.zeros(300)
#    r=np.arange(0,3,0.01)
#    r[299]=np.inf
#    res[bisect(r,x)]=1
#    return res
#
# t_tr_bis=np.array(list(map(conv,t_tr)))
#
# model=Sequential()
# model.add(Dense(units=50, activation='tanh',input_dim=len(f_tr[0])))
##model.add(Dropout(0.5))
##model.add(Dense(units=300, activation='tanh'))
##model.add(Dropout(0.5))
##model.add(Dense(units=300, activation='tanh'))
##model.add(Dropout(0.5))
# model.add(Dense(units=300, activation='softmax'))
# model.add(Lambda(lambda x: K.dot(x,K.transpose(K.constant(np.arange(0,3,0.01).reshape((1,300)))))))
#
# model.compile(loss='mse', optimizer='sgd')
# model.fit(f_tr,t_tr,epochs=15,batch_size=100)
#
##model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
##model.fit(f_tr,t_tr_bis,epochs=20,batch_size=50)
#
#
# print(model.predict(f_te),t_te,model.evaluate(f_te,t_te))

# model={}
# for i in list_user:
#    model[i]=ensemble.GradientBoostingRegressor()
#    model[i].fit(f_tr[user[i],:],t_tr[user[i]])

model = ensemble.GradientBoostingRegressor(n_estimators=100)
model.fit(f_tr, t_tr)
data = pd.read_csv('C:/Users/SAMUEL/Documents/MLC/test/data/datasets/supelec/test.csv')

f1 = pca1.transform(scale(data[list_eeg].as_matrix(), axis=0))
f2 = {}
for i in l:
    f2[i] = pca2[i].transform(scale(data[list_resp[i]].as_matrix(), axis=0))

f_test = np.concatenate([f1] + [f2[i] for i in l], axis=1)
f_test = np.concatenate([f_test, data[['time_previous', 'number_previous', 'time']].as_matrix(),
                         enc.transform(data['user'].as_matrix().reshape(50000, 1)).toarray()], axis=1)
# f_test=np.concatenate([f_test,data[['time_previous', 'number_previous', 'time']].as_matrix()],axis=1)

test_u = data['user'].as_matrix()

# f_test=np.concatenate([f_test,test_u.reshape(50000,1)],axis=1)

del (data)


def pred(x):
    if x[-1] in list_user:
        return model[x[-1]].predict(x[:-1].reshape(1, len(x) - 1))
    else:
        return 1 / len(list_user) * sum([model[u].predict(x[:-1]) for u in list_user])


# f_test=np.concatenate([f_test,f_test[:,-3:]],axis=1)

res = model.predict(f_test)
res = pd.DataFrame(res)
res.columns = ['power_increase']

res.to_csv('soumission6.csv')