# _main to run RNN analysis on target file
from RNN_decoder_class import *



############################################################
# IMPORT DATA
folder = 'data/'       #Specify path to folder containing DATA
file_name = 'N5_170929_No Obstacles_s_matrices.mat'

############################################################
## DATA variables:
bins_before = 10
neural_sig = 'APdat'            # Name of neural data
decoding_sig = 'EMGdat'         # Usually: 'EMGdat' / 'KINdat' (!!string)
decoding_labels = 'EMGlabels'   # Usually: 'EMGlabels' / 'KINlabels' (!!string) -> Leave as Empty string otherwise
signal = 1                      # EMG/Kinematic column to decode (FCR,FCU,ECR etc.)
allSignals = False               # If you want to analyze all signals (all EMGs/KINs) set this to TRUE. It won't matter what the variable 'signal' is then.

## NNet variables:
units = 50
dropout = 0
num_epochs = 10
verbose_flag = 1 #If you want to print the output during training

test_prop = 0.10 # The train_prop will automatically be 1-test_prop
valid_prop = 0  # Validation is used to test performance over training to avoid overfitting. Not implement yet, leave as 0.

do_folds = True
# If DO_FOLDS is True, the analysis will be done over the whole data length in
# n folds according to the train/test props: E.g. -> train = 0.9, test = 0.1 -> 10 folds
# If DO_FOLDS is False, the RNN will be trained on first 'train_prop' data, and tested on last 'test_prop' amount of data.

############################################################
# RUN RNN
############################################################
rnn = Rat_decoder(folder, bins_before, units, dropout, num_epochs)


rnn.import_data(file_name)

rnn.define_decoding_data(neural_sig, decoding_sig, decoding_labels)

# rnn.data_RNN_extraction(bins_before, signal)
#
# rnn.z_score_data()
#
# rnn.assign_test_val_sets(train_prop, test_prop, valid_prop)

rnn.train_model(test_prop, valid_prop, signal=signal, allSignals=allSignals, folds=False, verbose_flag=verbose_flag , mod=LSTM)

rnn.plot_results()
