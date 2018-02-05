# Code to:
# a. compute the kurtosis features of wave files
# b. operates on lists containing the wave file paths
# c. stores the features using HTK format
# ------
# Last updated: 10th July, 2017 (Neeks)
# Created by: Neeks
# ------

from scipy.io.wavfile import read as wavread
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import htkmfc as htk
import glob

# function definition
def get_kurtosis_(x,fs=16000,TWIN=0.025,THOP=0.01):

    # ----- get kurtosis
    nwin = int(TWIN*fs)
    hopsamps = int(THOP*fs)
    ncols = 1 + int((x.shape[0]-nwin)/hopsamps)
    Y = np.zeros((1,ncols),np.float)
    eps = np.spacing(np.float32(1.0))
    for i in range(ncols):
        temp = x[i*hopsamps + np.arange(0,nwin)]
        mu = np.mean(temp)
        mu4 = np.mean((temp-mu)**4)
        var = (np.mean((temp-mu)**2))**2
        if (var>eps):
            Y[0,i] = mu4/var
        else:
            Y[0,i] = eps
    return Y

# code to use the above function
wav_path = '/home/data/amicorpus/'; 
list_all_file_path ='/home/data/amicorpus/all_wave_path.list';

# all files list
with open(list_all_file_path) as f:
    all_files_with_path = f.read().splitlines()
print(all_files_with_path[0])

# training file list
list_path = '/home/neerajs/work/blurp_universe/AMI/Train_reduced.list'
with open(list_path) as f:
    useful_files = f.read().splitlines()
print(useful_files[0])

# feats storage path
feats_addr = '/home/neerajs/work/ami/feats/kurt/train/'

indx = 0
fullfilename=[]
filename=[]
for i in range(len(useful_files)):
    temp = useful_files[i].replace('_', '.')
    for j in range(len(all_files_with_path)):
        if((all_files_with_path[j].find(temp))!=-1):
            fullfilename.append(all_files_with_path[j])
            temp = useful_files[i].replace('.', '_')
            filename.append(temp)
            indx = indx + 1
            
nfiles = len(fullfilename)
for i in range(nfiles):
    temp = wav_path + fullfilename[i][2:]
    print(temp)
    [fs, x] = wavread(temp) # x is a numpy array of integer, representing the samples 
    # check for two channels
    if (x.ndim==2):
        x = x[:,0]
    # scale to -1.0 -- 1.0
    if x.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif x.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    x = x / (max_nb_bit + 1.0) 
    
    TWIN = 0.025
    THOP = 0.010
    KURT = get_kurtosis_(x,fs,TWIN,THOP)
    KURT = np.transpose(KURT)
    writer=htk.open(feats_addr+filename[i]+'.htk',mode='w',veclen=1) 
    writer.writeall(KURT)    
    writer.close()
