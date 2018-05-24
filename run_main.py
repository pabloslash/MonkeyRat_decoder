# _main to run RNN analysis on target file
from RNN_decoder_class import *



############################################################
# IMPORT DATA
folder = 'data/'       #Specify path to folder containing DATA
file_name = 'N6_171026_No Obstacles_s_matrices.mat'

############################################################
## DATA variables:
bins_before = 50
neural_sig = 'APdat'            # Name of neural data
decoding_sig = 'EMGdat'         # Usually: 'EMGdat' / 'KINdat' (!!string)
decoding_labels = 'EMGlabels'   # Usually: 'EMGlabels' / 'KINlabels' (!!string) -> Leave as Empty string otherwise
signal = 1                      # EMG/Kinematic column to decode (FCR,FCU,ECR etc.)
allSignals = True               # If you want to analyze all signals (all EMGs/KINs) set this to TRUE. It won't matter what the variable 'signal' is then.

## NNet variables:
units = 50
dropout = 0.1
num_epochs = 7
verbose_flag = 1 #If you want to print the output during training

test_prop = 0.10 # The train_prop will automatically be 1-test_prop
valid_prop = 0  # Validation is used to test performance over training to avoid overfitting. Not implement yet, leave as 0.

do_folds = True
# If DO_FOLDS is True, the analysis will be done over the whole data length in
# n folds according to the train/test props: E.g. -> train = 0.9, test = 0.1 -> 10 folds
# If DO_FOLDS is False, the RNN will be trained on first 'train_prop' data, and tested on last 'test_prop' amount of data.

# ############################################################
# # RUN RNN
# ############################################################
rnn = Rat_decoder(folder, bins_before, units, dropout, num_epochs)
rnn.import_data(file_name)
rnn.define_decoding_data(neural_sig, decoding_sig, decoding_labels)

rnn.train_model(test_prop, valid_prop, signal=signal, allSignals=allSignals, do_folds=do_folds, verbose_flag=verbose_flag , mod=LSTM)

rnn.plot_results(show_flag=True)


#
# ############################################################
# # Check dependence on units
# ############################################################
#
# epochs_v = [1, 5, 10, 15, 20]
# units_v = [20, 50, 75, 100, 150, 200, 500]
# bins_v = [5, 10, 20, 50, 100]
# dropout_v = [0.0, 0.1, 0.2, 0.4, 0.5]
#
#
# mean_vaf = []
#
# for u in xrange(len(dropout_v)):
#
#     dropout = dropout_v[u]
#
#     rnn = Rat_decoder(folder, bins_before, units, dropout, num_epochs)
#     rnn.import_data(file_name)
#     rnn.define_decoding_data(neural_sig, decoding_sig, decoding_labels)
#     rnn.train_model(test_prop, valid_prop, signal=signal, allSignals=allSignals, do_folds=do_folds, verbose_flag=verbose_flag , mod=LSTM)
#
#     mean_vaf.append(np.mean(rnn.vaf))
#
#
#
#
# plt.figure()
# plt.scatter(dropout_v, mean_vaf)
# plt.suptitle('N5_170929, muslce = {}'.format(rnn.EMGname))
# plt.title('Vaf vs. Dropout')
# plt.xlabel('# Dropout')
# plt.ylabel('vaf')
# plt.show(block=False)
