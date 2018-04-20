
# coding: utf-8

# ## Import libraries

# In[2]:

import theano as th
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
import sklearn.cross_validation
from scipy import io
from scipy import stats


# ## Import Data

# In[3]:

folder='data/'
data = io.loadmat(folder + 'N5_test2.mat')

X = data['X']
y = data['Y']


# ## Preprocess Data

# In[ ]:

#Normalize the inputs
X = stats.zscore(X,axis=0)

#Normalize the outputs
y_mean = np.mean(y) #Note I keep track of the original mean and stdev so we can put the predictions back in the original coordinates
y_std = np.std(y)
y = stats.zscore(y)


# In[ ]:

time = np.true_divide(np.arange(y.shape[0]+1),20) #Make a vector of times, for plotting later


# **Split into training/validation/testing sets**

# In[ ]:

#Train/test/valid proportions of data
train_prop=0.90
test_prop=0.5
valid_prop=0 #If you are seeing what parameters work best, you should do this on a separate validation set, to avoid overfitting to the test set

train_size=np.int(np.round(train_prop*X.shape[0]))
test_size=np.int(np.round(test_prop*X.shape[0]))

X_train=X[:train_size-X.shape[1],:,:] #Subtract X.shape[1] so we don't have overlap in the train/test sets
y_train=y[:train_size-X.shape[1],:]

X_test=X[train_size:test_size+train_size,:,:]
y_test=y[train_size:test_size+train_size,:]

X_valid=X[test_size+train_size:,:,:]
y_valid=y[test_size+train_size:,:]


time_train=time[:train_size]
time_test=time[train_size+1:]


# ## Testing Functions

# In[ ]:

#This function gets the VAF

def get_corr(y_test,y_test_pred):

    y_mean=np.mean(y_test)
    r2=1-np.sum((y_test_pred-y_test)**2)/np.sum((y_test-y_mean)**2)
    return r2


# ## RNN Decoding

# **Import packages**

# In[ ]:

#Import everything for keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, SimpleRNN, GRU
from keras.regularizers import l2, l1 #activity_l2, l1
from keras.callbacks import EarlyStopping


# **Code to run an RNN**

# In[ ]:

#Parameters
units=50
dropout=0
num_epochs=10
verbose_flag=1 #If you want to see the output during training


# In[ ]:

#Create model
model=Sequential()
#If you want to run an LSTM or GRU rather than a simpleRNN, in the next line change "SimpleRNN" to "GRU" or "LSTM"
model.add(SimpleRNN(units,input_shape=(X.shape[1],X.shape[2]),dropout_W=dropout,dropout_U=dropout))
if dropout!=0:
    model.add(Dropout(dropout))
model.add(Dense(1,init='uniform'))

#Compile model (includes object function and optimization technique)
model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])

#Fit model
model.fit(X_train,y_train,nb_epoch=num_epochs,verbose=verbose_flag)

#Get predictions on test set
y_test_pred = model.predict(X_test)
#Get VAF on test set
r2=get_corr(y_test=y_test,y_test_pred=y_test_pred)
print("vaf=", r2)


# **Plot fit**

# In[ ]:

#Rescale data to its original coordinates
y_test_rescale=(y_test*y_std)+y_mean
y_test_pred_rescale=(y_test_pred*y_std)+y_mean
#Plot
plt.figure()
plt.title('N5_170929 - vaf=%s' %str(r2))
plt.plot(time_test,y_test_rescale)
plt.plot(time_test,y_test_pred_rescale)
plt.xlabel('Time (s)')
plt.ylabel('EMG2')
plt.legend(['Actual','Predicted'],bbox_to_anchor=(1, .25))
# plt.ylim([0, 220])
plt.show(block=False)


# In[ ]:
