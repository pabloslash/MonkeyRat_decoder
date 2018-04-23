# _main to run RNN analysis on target file
from RNN_decoder_class import *



############################################################
# IMPORT DATA
folder = 'data/' #Specify path to DATA FOLDER
file_name = 'N5_170929_No Obstacles_s_matrices.mat'
# data = io.loadmat(folder + 'N5_170929_No Obstacles_s_matrices.mat')

############################################################
## Variables:
bins_before = 10
neural_sig = 'APdat'        # Name of neural data
decoding_sig = 'EMGdat'     # Usually: 'EMGdat' / 'KINdat' (!!string)
decoding_labels = 'EMGlabels'  # Usually: 'EMGlabels' / 'KINlabels' (!!string)
signal = 1                  # EMG/Kinematic column to decode (FCR,FCU,ECR etc.)

units = 50
dropout = 0
num_epochs = 10
verbose_flag = 1 #If you want to see the output during training

train_prop = 0.90
test_prop = 0.10
valid_prop = 0 #If you are seeing what parameters work best, you should do this on a separate validation set, to avoid overfitting to the test set

############################################################
# RUN RNN
############################################################
rnn_dec = Rat_decoder(folder)


rnn_dec.import_data(file_name)
rnn_dec.define_decoding_data(neural_sig, decoding_sig, decoding_labels)
rnn_dec.data_RNN_extraction(bins_before, signal)
rnn_dec.z_score_data()
rnn_dec.assign_test_val_sets(train_prop, test_prop, valid_prop)

rnn_dec.fit_model(units, dropout, num_epochs, verbose_flag, mod=LSTM)

rnn_dec.plot_results()
