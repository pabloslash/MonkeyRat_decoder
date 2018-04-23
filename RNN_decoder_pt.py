import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sklearn.cross_validation
import sklearn.model_selection
from scipy import io
from scipy import stats
import os
import seaborn as sns



def get_corr(y_test,y_test_pred):
    y_test = np.expand_dims(y_test,1)
    y_mean = np.mean(y_test)
    r2 = 1-np.sum((y_test_pred-y_test)**2)/np.sum((y_test-y_mean)**2)
    return r2

# #This function gets the VAF
# def get_corr(y_test,y_test_pred):
#
#     y_mean = np.mean(y_test)
#     r2 = 1-np.sum((y_test_pred-y_test)**2) / np.sum((y_test-y_mean)**2)
#     return r2

# Imported from cell as unicode in the last data I worked with.
# Might give errors in the future if not in the same format
def get_EMG_name(decoding_sig, signal, labels):
    EMGname = ''
    if 'EMG' in decoding_sig:
        EMGname = labels[0,signal][0].encode('utf8')
    elif 'KIN' in decoding_sig:
        EMGname = labels[0,signal][0].encode('utf8')
    return EMGname

def data_RNN_extraction():
    return 0

# IMPORT DATA
folder = 'data/'
# data = io.loadmat(folder + 'N5_170929_No Obstacles_s_matrices.mat')
data = io.loadmat(folder + 'N5_170929_No Obstacles_s_matrices.mat')

## LOAD DATA:
bins_before = 100
neural_sig = 'APdat'        # Name of neural data
decoding_sig = 'EMGdat'     # Usually: 'EMGdat' / 'KINdat' (!!string)
# decoding_labels = 'EMGlabels'  # Usually: 'EMGlabels' / 'KINlabels' (!!string)
signal = 1                  # EMG/Kinematic column to decode (FCR,FCU,ECR etc.)


neural_dat = data[neural_sig]
dec_sig_dat = data[decoding_sig][:]
# sigs_labels = data[decoding_labels]

num_neurons = neural_dat.shape[1]
num_ex = neural_dat.shape[0] - bins_before  # Predictions start after "bins_before" bins


# EMGname = get_EMG_name(decoding_sig, signal, sigs_labels)


# Prepare data to throw into RNN
y = dec_sig_dat[bins_before:bins_before+num_ex, signal]

# Make the covariate matrix for an RNN
# Each example (1st dimension) has "bins_before" bins of neural activity
# (2nd dimension) for all neurons (3rd dimension)
X = np.empty((num_ex,bins_before,num_neurons))
X[:] = np.nan
for i in xrange (num_ex):
    X[i,:,:] = neural_dat[i:i+bins_before,:]




# print(EMGname)

## Convert data

# y = y[:,muscle]
# io.savemat(folder + 'TestYY.mat', {'y':y})

plt.figure()
plt.plot(y)#, label='Validation Set, (Classification Accuracy) = %.2f%s' %(np.max(etr), '%'))
# plt.title(decoding_sig + ': ' + EMGname)
plt.xlabel('Bins')
#plt.ylabel('Percent Correct Classification')
# plt.legend(loc='lower right')
plt.show(block=False)



# PREPROCESS DATA
#Normalize the inputs
X = stats.zscore(X,axis=0)

#Normalize the outputs
y_mean = np.mean(y) #Note I keep track of the original mean and stdev so we can put the predictions back in the original coordinates
y_std = np.std(y)
y = stats.zscore(y)

print('Mean of decoding signal: ' + str(y_mean))
print('STD of decoding signal: ' + str(y_std))

#Make a vector of times, for plotting later
time = np.true_divide(np.arange(y.shape[0] + 1),20)


# Understand Data
X
X.shape[0]



# SPLIT INTO TRAINING/ VALIDATION/TESTING sets
#Train/test/valid proportions of data
train_prop=0.90
test_prop=0.1
valid_prop=0 #If you are seeing what parameters work best, you should do this on a separate validation set, to avoid overfitting to the test set

train_size = np.int(np.round(train_prop * X.shape[0]))
test_size = np.int(np.round(test_prop * X.shape[0]))

X_train = X[:train_size - X.shape[1],:,:] #Subtract X.shape[1] so we don't have overlap in the train/test sets
y_train = y[:train_size - X.shape[1]]

X_test = X[train_size:test_size + train_size,:,:]
y_test = y[train_size:test_size + train_size]

X_valid = X[test_size + train_size:,:,:]
y_valid = y[test_size + train_size:]


if valid_prop !=0:
    X_trueTest = X_test
    y_trueTest = y_test
    X_test = X_valid
    y_test = y_valid

time_train = time[:train_size]
time_test = time[train_size + 1:]

# Save variables to plot in MATLAB

saveResultsDir = (folder + 'PredResults/')

# saveResultsDir = (folder + 'PredResults/' + EMGname + '/')
# if not os.path.exists(saveResultsDir):
#     os.makedirs(saveResultsDir)

#Save train + test to check
io.savemat(saveResultsDir + 'TestTrain.mat', {'train_neural':X_train, 'test_neural':X_test, 'train_EMG':y_train, 'test_EMG':y_test})

#RNN DECODING
#Import everything for keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, SimpleRNN, GRU
from keras.regularizers import l2, l1 #activity_l2, l1
from keras.callbacks import EarlyStopping


#Parameters
units = 50
dropout = 0
num_epochs = 10
verbose_flag = 1 #If you want to see the output during training

#Create model
model = Sequential()
#If you want to run an LSTM or GRU rather than a simpleRNN, in the next line change "SimpleRNN" to "GRU" or "LSTM"
model.add(LSTM(units,input_shape=(X.shape[1],X.shape[2]),dropout_W=dropout,dropout_U=dropout))
if dropout!=0:
    model.add(Dropout(dropout))
model.add(Dense(1,init='uniform'))

#Compile model (includes object function and optimization technique)
model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])

#Fit model
model.fit(X_train,y_train,nb_epoch=num_epochs,verbose=verbose_flag)
# time.sleep(0.2)

#Get predictions on test set
y_test_pred = model.predict(X_test)
#Get VAF on test set
r2 = get_corr(y_test=y_test, y_test_pred=y_test_pred)
print("vaf=", r2)

#Only if there is validation
if valid_prop !=0:
    y_trueTest_pred = model.predict(X_trueTest)
    #Get VAF on test set
    r1 = get_corr(y_test=y_trueTest, y_test_pred=y_trueTest_pred)
    print("vaf2=", r1)

print("units=" , units)
print("dropout=" , dropout)
print("epochs=" , num_epochs)


# PLOT FIT
#Rescale data to its original coordinates
y_test_rescale = (y_test*y_std) + y_mean
y_test_pred_rescale = (y_test_pred*y_std) + y_mean

# r2 = get_corr(y_test_rescale, y_test_pred_rescale)
# print("vaf=", r2)

#Plot
plt.plot(time_test,y_test_rescale)
plt.plot(time_test,y_test_pred_rescale)
# plt.title(sMuscle)
plt.xlabel('Time (s)')
plt.ylabel('EMG')
plt.legend(['Actual','Predicted'],bbox_to_anchor=(1, .25))
plt.show(block=False)
#plt.ylim([0, 220])

# Save variables to plot in MATLAB
saveResultsDir = (folder + 'PredResults/' + EMGname + '/')
if not os.path.exists(saveResultsDir):
    os.makedirs(saveResultsDir)

#print (y_test_pred)
#print (y_test)

io.savemat(saveResultsDir + 'RNN_predResults.mat', {'muscle': EMGname, 'RNN_time': time_test, 'RNN_EMG':y_test_rescale, 'RNN_predEMG':y_test_pred_rescale, 'vaf':r2})
