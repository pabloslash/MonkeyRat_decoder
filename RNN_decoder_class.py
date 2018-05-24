import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline # For jupyter Notebooks
# import sklearn.cross_validation #Deprecated
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
import IPython as IP
#pyc files
import sys
sys.dont_write_bytecode = True


class Rat_decoder(object):
    def __init__(self, data_dir, bins_before, units, dropout, num_epochs):
        self.data_dir = data_dir
        self.save_dir = self.data_dir + 'PredResuls/'
        self.val_data = None
        self.bins_before = bins_before
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs

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
        if (self.sigs_labels.size > 0): #If we specified signal labels
            EMGname = self.sigs_labels[0,signal][0].encode('utf8')
        elif 'EMG' in self.decoding_sig:
            EMGname = 'EMG' + str(signal)
        elif 'KIN' in self.decoding_sig:
            EMGname = 'KIN' + str(signal)
        return EMGname
        # EMG/Kinematic column to decode (FCR,FCU,ECR etc.)

    def import_data(self, file_name):
        '''
        Will load NEURAL + DECODING sig (EMG/KIN) from .mat file with data stored in matrix form.
        '''
        folder = self.data_dir
        self.data = io.loadmat(folder + file_name)

        # Make directories to save files
        if not (os.path.isdir(self.save_dir)):
            os.mkdir(self.save_dir)

        self.animal_dir_save = (self.save_dir + file_name[:-4] + '/') # -4 to get rid of .mat extension
        if not (os.path.isdir(self.animal_dir_save)):
            os.mkdir(self.animal_dir_save)


    def define_decoding_data(self, neural_sig, decoding_sig, decoding_labels):
        self.decoding_sig = decoding_sig    # Name of decoding Sig ('EMG/KIN')
        self.neural_dat = self.data[neural_sig]
        self.dec_sig_dat = self.data[decoding_sig][:]
        if decoding_labels:
            self.sigs_labels = self.data[decoding_labels]   # Labels

    def data_RNN_extraction(self, sig):
        self.num_neurons = self.neural_dat.shape[1]
        self.num_ex = self.neural_dat.shape[0] - self.bins_before  # Predictions start after "bins_before" bins
        self.EMGname = self.get_EMG_name(sig)

        # Prepare data to throw into RNN
        self.y = self.dec_sig_dat[self.bins_before:self.bins_before + self.num_ex, sig]

        # Make the covariate matrix for an RNN
        # Each example (1st dimension) has "bins_before" bins of neural activity
        # (2nd dimension) for all neurons (3rd dimension)
        self.X = np.empty((self.num_ex, self.bins_before, self.num_neurons))
        self.X[:] = np.nan
        for i in xrange (self.num_ex):
            self.X[i,:,:] = self.neural_dat[i:i+self.bins_before,:]

        print('Decoding: ' + self.EMGname)

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


    def assign_test_val_sets(self, test_prop, valid_prop, fold):

        test_size = np.int(np.floor(test_prop * self.X.shape[0])) #Floor it to make sure we don't exceed dims of matrix
        test_begin = np.int( test_size * (fold+1) - test_size ) #If fold = 0, we start from the beginning.

        self.X_test = self.X[test_begin:test_begin + test_size,:,:]
        self.y_test = self.y[test_begin:test_begin + test_size]

        self.X_train = np.concatenate((self.X[:test_begin,:,:], self.X[test_begin+test_size:,:,:])) #Slice X_test from middle of data
        self.y_train = np.concatenate((self.y[:test_begin], self.y[test_begin+test_size:]))

        self.time_train = self.time[:self.X_train.shape[0]]
        # self.time_test = self.time[self.X_train.shape[0]+1:]


        # train_size = np.int(np.round(train_prop * self.X.shape[0]))
        # test_size = np.int(np.round(test_prop * self.X.shape[0]))

        # self.X_train = self.X[:train_size - self.X.shape[1],:,:] #Subtract X.shape[1] so we don't have overlap in the train/test sets
        # self.y_train = self.y[:train_size - self.X.shape[1]]

        # self.X_test = self.X[train_size:test_size + train_size,:,:]
        # self.y_test = self.y[train_size:test_size + train_size]

        # self.time_train = self.time[:train_size]
        # self.time_test = self.time[train_size + 1:]

        # # Get validation data out of X first if we want validation.
        # self.X_valid = self.X[test_size + train_size:,:,:]
        # self.y_valid = self.y[test_size + train_size:]

        # if valid_prop !=0:
        #     X_trueTest = X_test
        #     y_trueTest = y_test
        #     X_test = X_valid
        #     y_test = y_valid


    def train_model(self, test_prop, valid_prop, signal=0, allSignals=True, do_folds=False, verbose_flag=0, mod=LSTM):

        if (allSignals == True):
            signal = np.array([i for i in xrange(self.dec_sig_dat.shape[1])])
        else: signal = [signal]

        #Loop trhough all signals of interest
        for sig in signal:

            self.data_RNN_extraction(sig) # Get matrices X, y to train RNN
            self.z_score_data()           # Normalize data

            num_folds = np.int(np.round(1.0 / test_prop)) # These are the #folds we could split the data into with this test_proportion
            if (do_folds == True):
                test_prop = 1.0 / num_folds # Round test_prop according to num_folds
                folds = [f for f in xrange(num_folds)]
            else: folds = [num_folds - 1] # If we don't wanna do_folds, just take the last part of the data as testing and do 1 fold.

            #Variables to save / plot
            self.test_actual = np.array([])
            self.test_predicted = np.array([])
            self.vaf = np.array([])

            for fold in folds:

                self.assign_test_val_sets(test_prop, valid_prop, fold) #We only need the test_prop, the train data will be everything else

                #Create model
                model = Sequential()
                #If you want to run an LSTM or GRU rather than a simpleRNN, in the next line change "SimpleRNN" to "GRU" or "LSTM"
                model.add(mod(self.units,input_shape=(self.X.shape[1],self.X.shape[2]),dropout_W=self.dropout,dropout_U=self.dropout))
                if self.dropout!=0:
                    model.add(Dropout(self.dropout))
                model.add(Dense(1,init='uniform'))

                #Compile model (includes object function and optimization technique)
                model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])

                #Fit model
                model.fit(self.X_train,self.y_train,nb_epoch=self.num_epochs,verbose=verbose_flag)

                #Get predictions on test set
                self.y_test_pred = model.predict(self.X_test)
                #Get VAF on test set
                vaf = self.get_corr(y_test=self.y_test, y_test_pred=self.y_test_pred)

                print("Decoding {}".format(self.EMGname))
                print("Fold {} / {}: vaf = {}".format(fold+1,len(folds),vaf))

                #Variables to save / plot
                self.test_actual = np.append(self.test_actual, self.y_test, axis = 0)
                self.test_predicted = np.append(self.test_predicted, self.y_test_pred.squeeze(1), axis = 0)
                self.vaf = np.append(self.vaf, vaf)
                print (self.vaf)


            #######################################
            # Save results:
            signal_dir = (self.animal_dir_save + self.EMGname + '/')
            if not (os.path.isdir(signal_dir)):
                os.mkdir(signal_dir)

            io.savemat(signal_dir + self.EMGname + '_predictions.mat', {'y_actual':(self.test_actual*self.y_std)+self.y_mean,
                       'y_predicted':(self.test_predicted*self.y_std)+self.y_mean, 'EMG_name':self.EMGname, 'mean_vaf':np.mean(self.vaf), 'folds_VAFs':self.vaf})
            # io.savemat(signal_dir + 'folds_VAFs.mat', {'VAFs':self.vaf})

            self.plot_results(plot_range=[10], show_flag=False) #Plot and save 10s
            plt.savefig(signal_dir + 'prediction_fig.png')
            plt.close('all')


    def plot_results(self, plot_range=[], show_flag=False):
        # PLOT FIT
        #Rescale data to its original coordinates
        y_test_rescale = (self.test_actual * self.y_std) + self.y_mean
        y_test_pred_rescale = (self.test_predicted * self.y_std) + self.y_mean
        time = np.true_divide(np.arange(self.test_actual.shape[0]), 20)

        if (len(plot_range)==0):    #If no range in seconds is specfied, plot the whole signal
            plot_range = [time[-1]]

        #Plot
        plt.figure()
        plt.plot(time, y_test_rescale, 'blue')
        plt.plot(time, y_test_pred_rescale, 'orange')
        plt.title((self.EMGname + ': vaf = {}'.format(np.mean(self.vaf))))
        plt.xlabel('Time (s)')
        plt.xlim(xmin=0, xmax=plot_range[0])
        plt.ylabel(self.decoding_sig)
        plt.legend(['Actual','Predicted'], loc='upper right')
        if (show_flag):
            plt.show(block=False)
        #plt.ylim([0, 220])


        # # PLOT RAW EMG
        # plt.figure()
        # plt.plot(self.y)#, label='Validation Set, (Classification Accuracy) = %.2f%s' %(np.max(etr), '%'))
        # plt.title(self.decoding_sig + ': ' + self.EMGname)
        # plt.xlabel('Bins')
        # #plt.ylabel('Percent Correct Classification')
        # # plt.legend(loc='lower right')
        # plt.show(block=False)
