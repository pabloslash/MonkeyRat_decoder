import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline # For jupyter Notebooks
import sklearn.cross_validation
import sklearn.model_selection
from scipy import io
from scipy import stats
import os
import seaborn as sns
# Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, SimpleRNN, GRU
from keras.regularizers import l2, l1 #activity_l2, l1
from keras.callbacks import EarlyStopping


class Rat_decoder(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.val_data = None

    #This function gets the VAF
    def get_corr(self, y_test, y_test_pred):
        y_test = np.expand_dims(y_test,1)
        y_mean = np.mean(y_test)
        r2 = 1-np.sum((y_test_pred-y_test)**2)/np.sum((y_test-y_mean)**2)
        return r2

    # Imported from cell as unicode in the last data I worked with.
    # Might give errors in the future if not in the same format
    def get_EMG_name(self, signal):
        EMGname = ''
        if 'EMG' in self.decoding_sig:
            EMGname = self.sigs_labels[0,signal][0].encode('utf8')
        elif 'KIN' in self.decoding_sig:
            EMGname = self.sigs_labels[0,signal][0].encode('utf8')
        return EMGname
        # EMG/Kinematic column to decode (FCR,FCU,ECR etc.)

    def import_data(self, file_name):
        '''
        Will load NEURAL + DECODING sig (EMG/KIN) from .mat file with data stored in matrix form.
        '''
        folder = self.data_dir
        self.data = io.loadmat(folder + file_name)

    def define_decoding_data(self, neural_sig, decoding_sig, decoding_labels):
        self.decoding_sig = decoding_sig    # Name of decoding Sig ('EMG/KIN')
        self.sigs_labels = self.data[decoding_labels]   # Labels
        self.neural_dat = self.data[neural_sig]
        self.dec_sig_dat = self.data[decoding_sig][:]

    def data_RNN_extraction(self, bins_before, signal):
        self.num_neurons = self.neural_dat.shape[1]
        self.num_ex = self.neural_dat.shape[0] - bins_before  # Predictions start after "bins_before" bins
        self.EMGname = self.get_EMG_name(signal)

        # Prepare data to throw into RNN
        self.y = self.dec_sig_dat[bins_before:bins_before + self.num_ex, signal]

        # Make the covariate matrix for an RNN
        # Each example (1st dimension) has "bins_before" bins of neural activity
        # (2nd dimension) for all neurons (3rd dimension)
        self.X = np.empty((self.num_ex, bins_before, self.num_neurons))
        self.X[:] = np.nan
        for i in xrange (self.num_ex):
            self.X[i,:,:] = self.neural_dat[i:i+bins_before,:]

        print('EMG: ' + self.EMGname)

        ## Convert data

        # y = y[:,muscle]
        # io.savemat(folder + 'TestYY.mat', {'y':y})

        plt.figure()
        plt.plot(self.y)#, label='Validation Set, (Classification Accuracy) = %.2f%s' %(np.max(etr), '%'))
        plt.title(self.decoding_sig + ': ' + self.EMGname)
        plt.xlabel('Bins')
        #plt.ylabel('Percent Correct Classification')
        # plt.legend(loc='lower right')
        plt.show(block=False)


    def z_score_data(self):

        # PREPROCESS DATA
        #Normalize the inputs
        self.X = stats.zscore(self.X,axis=0)

        #Normalize the outputs
        self.y_mean = np.mean(self.y) #Note I keep track of the original mean and stdev so we can put the predictions back in the original coordinates
        self.y_std = np.std(self.y)
        self.y = stats.zscore(self.y)

        print('Mean of decoding signal: ' + str(self.y_mean))
        print('STD of decoding signal: ' + str(self.y_std))

        #Make a vector of times, for plotting later
        self.time = np.true_divide(np.arange(self.y.shape[0] + 1),20)

        # Understand Data
        self.X.shape[0]


    def assign_test_val_sets(self, train_prop, test_prop, valid_prop):
        train_size = np.int(np.round(train_prop * self.X.shape[0]))
        test_size = np.int(np.round(test_prop * self.X.shape[0]))

        self.X_train = self.X[:train_size - self.X.shape[1],:,:] #Subtract X.shape[1] so we don't have overlap in the train/test sets
        self.y_train = self.y[:train_size - self.X.shape[1]]

        self.X_test = self.X[train_size:test_size + train_size,:,:]
        self.y_test = self.y[train_size:test_size + train_size]

        self.X_valid = self.X[test_size + train_size:,:,:]
        self.y_valid = self.y[test_size + train_size:]

        if valid_prop !=0:
            X_trueTest = X_test
            y_trueTest = y_test
            X_test = X_valid
            y_test = y_valid

        self.time_train = self.time[:train_size]
        self.time_test = self.time[train_size + 1:]


    def fit_model(self, units, dropout, num_epochs, verbose_flag, mod=LSTM):
        #Create model
        model = Sequential()
        #If you want to run an LSTM or GRU rather than a simpleRNN, in the next line change "SimpleRNN" to "GRU" or "LSTM"
        model.add(mod(units,input_shape=(self.X.shape[1],self.X.shape[2]),dropout_W=dropout,dropout_U=dropout))
        if dropout!=0:
            model.add(Dropout(dropout))
        model.add(Dense(1,init='uniform'))

        #Compile model (includes object function and optimization technique)
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])

        #Fit model
        model.fit(self.X_train,self.y_train,nb_epoch=num_epochs,verbose=verbose_flag)

        #Get predictions on test set
        self.y_test_pred = model.predict(self.X_test)
        #Get VAF on test set
        r2 = self.get_corr(y_test=self.y_test, y_test_pred=self.y_test_pred)
        print("vaf=", r2)

        #
        #
        # #Only if there is validation
        # if valid_prop !=0:
        #     y_trueTest_pred = model.predict(X_trueTest)
        #     #Get VAF on test set
        #     r1 = get_corr(y_test=y_trueTest, y_test_pred=y_trueTest_pred)

    def plot_results(self):
        # PLOT FIT
        #Rescale data to its original coordinates
        y_test_rescale = (self.y_test * self.y_std) + self.y_mean
        y_test_pred_rescale = (self.y_test_pred * self.y_std) + self.y_mean

        #Plot
        plt.plot(self.time_test, y_test_rescale)
        plt.plot(self.time_test, y_test_pred_rescale)
        # plt.title(sMuscle)
        plt.xlabel('Time (s)')
        plt.ylabel('EMG')
        plt.legend(['Actual','Predicted'],bbox_to_anchor=(1, .25))
        plt.show(block=False)
        #plt.ylim([0, 220])
