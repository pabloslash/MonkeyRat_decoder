# _main to run RNN analysis on target file


# IMPORT DATA
folder = 'data/'
file_name = 'N5_170929_No Obstacles_s_matrices.mat'
# data = io.loadmat(folder + 'N5_170929_No Obstacles_s_matrices.mat')

## LOAD DATA:
bins_before = 10
neural_sig = 'APdat'        # Name of neural data
decoding_sig = 'EMGdat'     # Usually: 'EMGdat' / 'KINdat' (!!string)
decoding_labels = 'EMGlabels'  # Usually: 'EMGlabels' / 'KINlabels' (!!string)
signal = 1                  # EMG/Kinematic column to decode (FCR,FCU,ECR etc.)


## Instantiate class:
rnn_dec = Rat_decoder(folder)

# SPLIT INTO TRAINING/ VALIDATION/TESTING sets
#Train/test/valid proportions of data
train_prop = 0.90
test_prop = 0.10
valid_prop = 0 #If you are seeing what parameters work best, you should do this on a separate validation set, to avoid overfitting to the test set
